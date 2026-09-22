"""A especie `shot`: o instante do contato, que ancora a janela em que as outras especies medem."""

from edge.modules.shot.kinematics import BallBreakShot, observacoes
from edge.modules.shot.tdeed import TdeedShot

__all__ = ["BallBreakShot", "TdeedShot", "observacoes"]
