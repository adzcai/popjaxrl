from .meta_cartpole import NoisyStatelessMetaCartPole
from .popgym_autoencode import AutoencodeEasy, AutoencodeHard, AutoencodeMedium
from .popgym_battleship import BattleshipEasy, BattleshipHard, BattleshipMedium
from .popgym_cartpole import (
    NoisyStatelessCartPoleEasy,
    NoisyStatelessCartPoleHard,
    NoisyStatelessCartPoleMedium,
    StatelessCartPoleEasy,
    StatelessCartPoleHard,
    StatelessCartPoleMedium,
)
from .popgym_concentration import (
    ConcentrationEasy,
    ConcentrationHard,
    ConcentrationMedium,
)
from .popgym_count_recall import CountRecallEasy, CountRecallHard, CountRecallMedium
from .popgym_higherlower import HigherLowerEasy, HigherLowerHard, HigherLowerMedium
from .popgym_minesweeper import MineSweeperEasy, MineSweeperHard, MineSweeperMedium
from .popgym_multiarmedbandit import (
    MultiarmedBanditEasy,
    MultiarmedBanditHard,
    MultiarmedBanditMedium,
)
from .popgym_pendulum import (
    NoisyStatelessPendulumEasy,
    NoisyStatelessPendulumHard,
    NoisyStatelessPendulumMedium,
    StatelessPendulumEasy,
    StatelessPendulumHard,
    StatelessPendulumMedium,
)
from .popgym_repeat_first import RepeatFirstEasy, RepeatFirstHard, RepeatFirstMedium
from .popgym_repeat_previous import (
    RepeatPreviousEasy,
    RepeatPreviousHard,
    RepeatPreviousMedium,
)

__all__ = [
    # Meta CartPole
    "NoisyStatelessMetaCartPole",
    # Autoencode
    "AutoencodeEasy",
    "AutoencodeHard",
    "AutoencodeMedium",
    # Battleship
    "BattleshipEasy",
    "BattleshipHard",
    "BattleshipMedium",
    # CartPole
    "NoisyStatelessCartPoleEasy",
    "NoisyStatelessCartPoleHard",
    "NoisyStatelessCartPoleMedium",
    "StatelessCartPoleEasy",
    "StatelessCartPoleHard",
    "StatelessCartPoleMedium",
    # Concentration
    "ConcentrationEasy",
    "ConcentrationHard",
    "ConcentrationMedium",
    # Count Recall
    "CountRecallEasy",
    "CountRecallHard",
    "CountRecallMedium",
    # Higher Lower
    "HigherLowerEasy",
    "HigherLowerHard",
    "HigherLowerMedium",
    # Minesweeper
    "MineSweeperEasy",
    "MineSweeperHard",
    "MineSweeperMedium",
    # Multiarmed Bandit
    "MultiarmedBanditEasy",
    "MultiarmedBanditHard",
    "MultiarmedBanditMedium",
    # Pendulum
    "NoisyStatelessPendulumEasy",
    "NoisyStatelessPendulumHard",
    "NoisyStatelessPendulumMedium",
    "StatelessPendulumEasy",
    "StatelessPendulumHard",
    "StatelessPendulumMedium",
    # Repeat First
    "RepeatFirstEasy",
    "RepeatFirstHard",
    "RepeatFirstMedium",
    # Repeat Previous
    "RepeatPreviousEasy",
    "RepeatPreviousHard",
    "RepeatPreviousMedium",
]
