# Initialize a network such that:
# Normal power injections
# Chen: Remain within the line limit and make it lower artificially using a multiplicative factor
# For TP: re-run OPF on new network (to make fair comparison)

class Statistics:
    def __init__(self):
        self.lost_load: float = 0
        self.n_line_failures: int = 0


def cascading_failure(net):
    stats = Statistics()

    while not stop(net):
        # Sample a single line or two lines
        # Remove the line(s)
        # Re-run PF
        # Collect stats

    return stats

def stop(net):
    # Measure: lost load until no more line will be failing/overloaded, number of failed lines
    pass
