import numpy as np

from .array_utils import wrapping_array
from .game_physics import CellTypes


@np.vectorize
def render_cell(cell, goal=None, pristine=False, orientation=0):
    if goal is None:
        val = '\x1b[0m'
    elif goal == 0:
        val = '\x1b[48;5;7m '
    elif goal < 0 and pristine:
        val = '\x1b[48;5;211m '
    elif goal < 0:
        val = '\x1b[48;5;175m '
    elif pristine:
        val = '\x1b[48;5;44m '
    else:
        val = '\x1b[48;5;116m '
    val += {
        0: '\x1b[38;5;0m',
        CellTypes.color_r: '\x1b[38;5;1m',
        CellTypes.color_g: '\x1b[38;5;2m',
        CellTypes.color_b: '\x1b[38;5;12m',
        CellTypes.color_r | CellTypes.color_g: '\x1b[38;5;11m',
        CellTypes.color_g | CellTypes.color_b: '\x1b[38;5;39m',
        CellTypes.color_r | CellTypes.color_b: '\x1b[38;5;129m',
        CellTypes.rainbow_color: '\x1b[38;5;8m',
    }[cell & CellTypes.rainbow_color]

    SPRITES = {
        CellTypes.agent: '\x1b[1m' + '⋀>⋁<'[orientation],
        CellTypes.spawning: 'S',
        CellTypes.level_exit: 'X',
        CellTypes.plant: '&',
        CellTypes.ice_cube: '=',
        CellTypes.alive: 'z',
        CellTypes.crate: '%',
        CellTypes.wall: '#',
    }
    for sprite_val, sprite in SPRITES.items():
        # This isn't exactly fast, but oh well.
        if (cell & sprite_val) == sprite_val:
            val += sprite
            break
    else:
        val += '?' if cell else ' '
    val += '\x1b[0m'
    return val


def render_board(s, centered_view=False, view_size=None, fixed_orientation=False):
    """
    Renders the game state `s`. Does not include scores, etc.

    This is not exactly a speedy rendering system, but it should be plenty
    fast enough for our purposes.

    Parameters
    ----------
    view_size : (int width, int height)
        If not None, specifies the size of the view centered on the agent.
    fixed_orientation : bool
        If true, the board is re-oriented such that the player is always
        facing up.
    """
    if centered_view or view_size or fixed_orientation:
        if view_size is None:
            view_size = s.board.shape
        if fixed_orientation and s.orientation % 2 == 1:
            # transpose the view
            view_height, view_width = view_size
        else:
            view_width, view_height = view_size
        x0, y0 = s.agent_loc
        x0 -= view_width // 2
        y0 -= view_height // 2
        board = s.board.view(wrapping_array)[y0:y0+view_height, x0:x0+view_width]
        goals = s.goals.view(wrapping_array)[y0:y0+view_height, x0:x0+view_width]
        prior_states = s.prior_states.view(wrapping_array)[y0:y0+view_height, x0:x0+view_width]
    else:
        view_width, view_height = s.width, s.height
        board = s.board
        goals = s.goals
        prior_states = s.prior_states
    pristine = prior_states < 3
    screen = np.empty((view_height+2, view_width+3), dtype=object)
    screen[:] = ''
    screen[0] = screen[-1] = ' -'
    screen[:,0] = screen[:,-2] = ' |'
    screen[:,-1] = '\n'
    screen[0,0] = screen[0,-2] = screen[-1,0] = screen[-1,-2] = ' +'
    if fixed_orientation and s.orientation != 0:
        cells = render_cell(board, goals, pristine).view(np.ndarray)
        if s.orientation == 1:
            cells = cells.T[::-1]
        elif s.orientation == 2:
            cells = cells[::-1, ::-1]
        elif s.orientation == 3:
            cells = cells.T[:, ::-1]
        else:
            raise RuntimeError("Unexpected orientation: %s" % (s.orientation,))
        screen[1:-1,1:-2] = cells
    else:
        screen[1:-1,1:-2] = render_cell(board, goals, pristine, s.orientation)
    return ''.join(screen.ravel())
