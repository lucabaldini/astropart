# Copyright (C) 2023 luca.baldini@pi.infn.it
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""System-wide facilities.
"""

from pathlib import Path
import sys

from loguru import logger

# Logger setup.
DEFAULT_LOGURU_HANDLER = dict(sink=sys.stderr, colorize=True, format=">>> <level>{message}</level>")
logger.remove()
logger.add(**DEFAULT_LOGURU_HANDLER)

__pkgname__ = 'astropart'
__url__ = f'https://bitbucket.org/lbaldini/{__pkgname__}'

# Basic package structure.
ASTROPART_ROOT = Path(__file__).parent
ASTROPART_BASE = ASTROPART_ROOT.parent
ASTROPART_DATA = ASTROPART_ROOT / 'data'
ASTROPART_LATEX = ASTROPART_BASE / 'latex'
ASTROPART_MACRO = ASTROPART_LATEX / 'macro'
ASTROPART_FIGURES = ASTROPART_LATEX / 'figures'

# Path to the Python module containing the version information.
#ASTROPART_VERSION_FILE_PATH = ASTROPART_ROOT / '_version.py'

# Path to the release notes.
#ASTROPART_RELEASE_NOTES_PATH = ASTROPART_DOCS / 'release.rst'
