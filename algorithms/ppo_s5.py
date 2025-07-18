from collections.abc import Sequence
from typing import Any, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import spaces

from envs.wrappers import LogWrapper

from .s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO


class ActorCriticS5(nn.Module):
    action_dim: Sequence[int]
    config: dict
    ssm_init_fn: Any

    def setup(self):
        self.encoder_0 = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.encoder_1 = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )

        self.action_body_0 = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.action_body_1 = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.action_decoder = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

        self.value_body_0 = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.value_body_1 = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.value_decoder = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

        self.s5 = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.config["S5_D_MODEL"],
            n_layers=self.config["S5_N_LAYERS"],
            activation=self.config["S5_ACTIVATION"],
            do_norm=self.config["S5_DO_NORM"],
            prenorm=self.config["S5_PRENORM"],
            do_gtrxl_norm=self.config["S5_DO_GTRXL_NORM"],
        )
        if self.config["CONTINUOUS"]:
            self.log_std = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )

    def __call__(self, hidden, x):
        obs, dones = x
        if self.config.get("NO_RESET"):
            dones = jnp.zeros_like(dones)
        embedding = self.encoder_0(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = self.encoder_1(embedding)
        embedding = nn.leaky_relu(embedding)

        hidden, embedding = self.s5(hidden, embedding, dones)

        actor_mean = self.action_body_0(embedding)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_body_1(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_decoder(actor_mean)

        if self.config["CONTINUOUS"]:
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = self.value_body_0(embedding)
        critic = nn.leaky_relu(critic)
        critic = self.value_body_1(critic)
        critic = nn.leaky_relu(critic)
        critic = self.value_decoder(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = config["ENV"], config["ENV_PARAMS"]
    env = LogWrapper(env)

    config["CONTINUOUS"] = type(env.action_space(env_params)) == spaces.Box

    d_model = config["S5_D_MODEL"]
    ssm_size = config["S5_SSM_SIZE"]
    n_layers = config["S5_N_LAYERS"]
    blocks = config["S5_BLOCKS"]
    block_size = int(ssm_size / blocks)

    Lambda, _, _, V, _ = make_DPLR_HiPPO(ssm_size)
    block_size = block_size // 2
    ssm_size = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vinv = V.conj().T

    ssm_init_fn = init_S5SSM(
        H=d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init="lecun_normal",
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
        conj_sym=True,
        clip_eigs=False,
        bidirectional=False,
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if config["CONTINUOUS"]:
            network = ActorCriticS5(
                env.action_space(env_params).shape[0],
                config=config,
                ssm_init_fn=ssm_init_fn,
            )
        else:
            network = ActorCriticS5(
                env.action_space(env_params).n, config=config, ssm_init_fn=ssm_init_fn
            )
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = StackedEncoderModel.initialize_carry(
            config["NUM_ENVS"], ssm_size, n_layers
        )
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        partial_reset = lambda x: env.reset(x, env_params)
        obsv, env_state = jax.vmap(partial_reset)(reset_rng)
        # init_hstate = SequenceLayer.initialize_carry(config["NUM_ENVS"], ssm_size)
        init_hstate = StackedEncoderModel.initialize_carry(
            config["NUM_ENVS"], ssm_size, n_layers
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                # transition = Transition(done, action, value, reward, log_prob, last_obs, info, last_done)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    # delta = reward + config["GAMMA"] * next_value - value
                    # gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            if config["DEBUG"]:
                metric = (
                    traj_batch.info["return_info"][..., 1]
                    * traj_batch.info["returned_episode"]
                ).sum() / traj_batch.info["returned_episode"].sum()
                if config.get("LOG"):

                    def callback(metric):
                        print(metric)
                        wandb.log({"metric": metric})

                else:

                    def callback(metric):
                        print(metric)

                jax.debug.callback(callback, metric)
            else:
                metric = (
                    traj_batch.info["return_info"][..., 1]
                    * traj_batch.info["returned_episode"]
                ).sum() / traj_batch.info["returned_episode"].sum()

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return runner_state, metric

    return train


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "MemoryChain-bsuite",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "S5_D_MODEL": 256,
        "S5_SSM_SIZE": 256,
        "S5_N_LAYERS": 4,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": True,
        "S5_PRENORM": True,
        "S5_DO_GTRXL_NORM": True,
    }

    jit_train = jax.jit(make_train(config))

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
