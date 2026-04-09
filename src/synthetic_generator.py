import torch as to


def naive_data_generator(
    state_sequence: list, lengths: list, means: to.Tensor, variances: to.Tensor
) -> to.Tensor:
    """Generates datasets with independent variables jumping between different Gaussians

    Args:
        state_sequence (list): Sequence of hidden states
        lengths (list): Sequence of lengths
        means (to.Tensor): means for each hidden states [nstates, nvariables]
        variances (to.Tensor): variances for each hidden states [nstates, nvariables]

    Returns:
        to.Tensor: generated data
    """
    assert len(state_sequence) == len(lengths)
    assert to.min(variances) > 1e-5
    data = []
    nsegments = len(state_sequence)
    for i in range(nsegments):
        segmentlen = lengths[i]
        state = state_sequence[i]
        statemean = means[state]
        statevar = variances[state]
        data.append(
            to.stack([to.normal(statemean, statevar) for _ in range(segmentlen)])
        )
    return to.cat(data, dim=0)


def mvn_data_generator(
    state_sequence: list, lengths: list, means: to.Tensor, variances: to.Tensor
) -> to.Tensor:
    """Generates datasets assuming multivariate normal distribution.
    Data jumps between different Gaussians

    Args:
        state_sequence (list): Sequence of hidden states
        lengths (list): Sequence of lengths
        means (to.Tensor): means for each hidden states [nstates, nvariables]
        variances (to.Tensor): covariances for each hidden states [nstates, nvariables, nvariables]

    Returns:
        to.Tensor: generated data
    """
    assert len(state_sequence) == len(lengths)
    assert to.min(variances) > 1e-5
    data = []
    nsegments = len(state_sequence)
    generators = []
    nstates = len(means)
    for i in range(nstates):
        statemean = means[i]
        statevar = variances[i]
        generators.append(to.distributions.MultivariateNormal(statemean, statevar))
    for i in range(nsegments):
        segmentlen = lengths[i]
        state = state_sequence[i]
        statemean = means[state]
        statevar = variances[state]
        data.append(generators[state].sample((segmentlen,)))
    return to.cat(data, dim=0)
