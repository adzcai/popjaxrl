from collections.abc import Callable, Sequence
import math
from typing import NamedTuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from gymnax.environments import spaces
from jax.nn import initializers
from jaxtyping import Array, Bool, Float, Key, PyTree

from envs.wrappers import LogWrapper


Initializer = Callable[[Key[Array, ""], tuple[int, ...], jax.typing.DTypeLike], Array]


def tree_initialize(
    get_weights: Callable[[PyTree], Sequence[Array]],
    model: PyTree,
    initializers: Sequence[Initializer],
    *,
    key: Key[Array, ""],
):
    """Initialize parameters of an Equinox module using jax.nn.initializers."""
    weights = get_weights(model)
    weights = [
        init(k, weight.shape, weight.dtype)
        for weight, init, k in zip(
            weights,
            initializers,
            jr.split(key, len(weights)),
            strict=True,
        )
    ]
    return eqx.tree_at(get_weights, model, weights)


def mlp_orthogonal(model: eqx.nn.MLP, scales: Sequence[float], key: Key[Array, ""]):
    return tree_initialize(
        lambda net: [linear.weight for linear in net.layers]
        + [linear.bias for linear in net.layers],
        model,
        [initializers.orthogonal(scale) for scale in scales]
        + [initializers.zeros] * (model.depth + 1),
        key=key,
    )


class ScannedRNN(eqx.Module):
    cell: eqx.nn.GRUCell

    def __init__(self, hidden_size: int, input_size: int, *, key: Key[Array, ""]):
        cell = eqx.nn.GRUCell(
            hidden_size=hidden_size, input_size=input_size, key=jr.key(0)
        )
        self.cell = tree_initialize(
            lambda cell: [cell.weight_ih, cell.weight_hh, cell.bias, cell.bias_n],
            cell,
            [
                lambda key, shape, dtype: jnp.concat(
                    [
                        initializers.lecun_normal()(k, (shape[0] // 3, shape[1]), dtype)
                        for k in jr.split(key, 3)
                    ]
                ),
                lambda key, shape, dtype: jnp.concat(
                    [
                        initializers.orthogonal()(k, (shape[0] // 3, shape[1]), dtype)
                        for k in jr.split(key, 3)
                    ]
                ),
                initializers.zeros,
                initializers.zeros,
            ],
            key=key,
        )

    def __call__(
        self,
        rnn_state_tm1: Float[Array, " hidden_dim"],
        embedding_t: Float[Array, " obs_dim"],
        is_first_t: Bool[Array, ""],
    ):
        """Applies the module."""
        rnn_state_tm1 = jnp.where(is_first_t, self.initialize_carry(), rnn_state_tm1)
        rnn_state_t = self.cell(hidden=rnn_state_tm1, input=embedding_t)
        return rnn_state_t, rnn_state_t

    def initialize_carry(self) -> Float[Array, " hidden_size"]:
        return jnp.zeros(self.cell.hidden_size)


class ActorCriticRNN(eqx.Module):
    obs_embedding: eqx.nn.MLP
    actor_mean: eqx.nn.MLP
    critic: eqx.nn.MLP
    scanned_rnn: ScannedRNN
    action_log_std: Float[Array, " action_dim"] | None

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        hidden_size: int,
        action_size: int,
        continuous: bool,
        *,
        key: Key[Array, ""],
    ):
        key_obs, key_rnn, key_actor, key_critic = jr.split(key, 4)
        obs_embedding = eqx.nn.MLP(
            in_size=obs_shape[-1],
            out_size=256,
            width_size=128,
            depth=1,
            activation=jax.nn.leaky_relu,
            final_activation=jax.nn.leaky_relu,
            key=jr.key(0),
        )
        self.obs_embedding = mlp_orthogonal(
            obs_embedding, [math.sqrt(2)] * (obs_embedding.depth + 1), key=key_obs
        )
        self.scanned_rnn = ScannedRNN(
            hidden_size=hidden_size, input_size=256, key=key_rnn
        )
        actor_mean = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=action_size,
            width_size=128,
            depth=2,
            activation=jax.nn.leaky_relu,
            final_activation=jax.nn.leaky_relu,
            key=jr.key(0),
        )
        self.actor_mean = mlp_orthogonal(
            actor_mean,
            [2.0] * actor_mean.depth + [0.01],
            key=key_actor,
        )
        self.action_log_std = jnp.zeros(action_size) if continuous else None
        critic = eqx.nn.MLP(
            in_size=hidden_size,
            out_size="scalar",
            width_size=128,
            depth=2,
            activation=jax.nn.leaky_relu,
            key=jr.key(0),
        )
        self.critic = mlp_orthogonal(
            critic,
            [2.0] * critic.depth + [1.0],
            key=key_critic,
        )

    def __call__(
        self,
        rnn_state_tm1: Float[Array, " hidden_size"],
        obs_s: Float[Array, " horizon *obs_shape"],
        is_first_s: Bool[Array, " horizon"],
    ):
        embedding_s = jax.vmap(self.obs_embedding)(obs_s)
        rnn_state, hidden_s = jax.lax.scan(
            lambda carry, x: self.scanned_rnn(carry, *x),
            rnn_state_tm1,
            (embedding_s, is_first_s),
        )

        actor_mean = jax.vmap(self.actor_mean)(hidden_s)

        if self.action_log_std is not None:
            pi = distrax.MultivariateNormalDiag(
                actor_mean, jnp.exp(self.action_log_std)
            )
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = jax.vmap(self.critic)(hidden_s)
        return rnn_state, pi, critic


class Transition(NamedTuple):
    done: Bool[Array, ""]
    action: jnp.ndarray
    value: Float[Array, ""]
    reward: jnp.ndarray
    log_prob: Float[Array, ""]
    obs: Float[Array, " *obs_shape"]
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = config["ENV"], config["ENV_PARAMS"]
    env = LogWrapper(env)

    config["CONTINUOUS"] = type(env.action_space(env_params)) is spaces.Box

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        rng, _rng = jr.split(rng)
        action_size = (
            env.action_space(env_params).shape[0]
            if config["CONTINUOUS"]
            else env.action_space(env_params).n
        )
        network = ActorCriticRNN(
            obs_shape=env.observation_space(env_params).shape,
            hidden_size=256,
            action_size=action_size,
            continuous=config["CONTINUOUS"],
            key=_rng,
        )
        network_params, network_static = eqx.partition(network, eqx.is_inexact_array)
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

        def apply_fn(params, rnn_state_tm1, x):
            obs_t, is_first_t = x
            net = eqx.combine(params[0], network_static)
            rnn_state_bt, pi_bs, actor_bs = jax.vmap(net)(
                rnn_state_tm1, jnp.swapaxes(obs_t, 0, 1), jnp.swapaxes(is_first_t, 0, 1)
            )
            return (
                rnn_state_bt,
                type(pi_bs)(logits=jnp.swapaxes(pi_bs.logits, 0, 1)),
                jnp.swapaxes(actor_bs, 0, 1),
            )

        train_state = TrainState.create(
            apply_fn=apply_fn,
            params=(network_params,),  # must be container
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jr.split(rng)
        reset_rng = jr.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = jnp.tile(
            network.scanned_rnn.initialize_carry(), (config["NUM_ENVS"], 1)
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jr.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = apply_fn(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jr.split(rng)
                rng_step = jr.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
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
            _, _, last_val = apply_fn(train_state.params, hstate, ac_in)
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
                        _, pi, value = apply_fn(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
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

                rng, _rng = jr.split(rng)
                permutation = jr.permutation(_rng, config["NUM_ENVS"])
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

            init_hstate = initial_hstate[None, :]  # TBH
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

        rng, _rng = jr.split(rng)
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
