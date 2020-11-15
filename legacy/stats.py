
import pstats
from pstats import SortKey

filename = "profileresults"

with open(f'profiles/stats', 'w') as stream:
    stats = pstats.Stats(f'profiles/{filename}', stream=stream)
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(100)