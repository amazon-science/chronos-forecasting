import numpy as np
from gluonts.dataset.artificial import recipe as rcp


def make_covariate_gen(covariate_func):
    def covariate_gen(series, op, function_target, **kwargs):
        covariates = covariate_func(
            series=series, op=op, function_target=function_target, **kwargs
        )

        if op == "add":
            return covariates + series, covariates
        elif op == "mult":
            return np.multiply(covariates, series), covariates
        else:
            raise NameError()

    return covariate_gen


@make_covariate_gen
def steps(
    series: np.ndarray,
    op: str,
    function_target: str,
    num_events: int,
    delta: int,
    fixed_event_pos: int = None,
) -> np.ndarray:

    length = len(series)
    step_pos = np.random.randint(0, length, num_events)

    if fixed_event_pos is not None:
        step_pos[-1] = length - fixed_event_pos

    step_delta = np.random.randint(1, delta, num_events)
    steps_series = np.ones(length)
    gamma = get_gamma(series, function_target, op)

    for p, delta in zip(step_pos, step_delta):
        steps_series[p : p + delta] = gamma

    return steps_series


@make_covariate_gen
def bells(
    series: np.ndarray,
    op: str,
    function_target: str,
    num_events: int,
    max_sigma: int = 15,
    fixed_event_pos: int = None,
):

    length = len(series)
    change_pos = np.random.randint(0, length, num_events)
    sigma_change = np.random.uniform(1, max_sigma, num_events).reshape(-1, 1)

    if fixed_event_pos is not None:
        change_pos[-1] = length - fixed_event_pos
        sigma_change[-1] = 7

    gamma = get_gamma(series, function_target, op)
    t = np.tile(np.arange(length).reshape(1, -1), (num_events, 1))
    change_pos = np.tile(change_pos.reshape(-1, 1), (1, length))

    bells_series = gamma * (
        np.exp(-np.divide(t - change_pos, sigma_change) ** 2).sum(axis=0) + 1
    )

    return bells_series


@make_covariate_gen
def arp(series: np.ndarray, op: str, function_target: str):
    phi1 = np.random.uniform(0, 1)
    phi2 = 1 - phi1

    # AutoRegressive Process
    target = rcp.ARp(phi=[phi1, phi2], sigma=1)
    x = rcp.evaluate(dict(target=target), length=len(series))["target"]
    x = (x - x.min()) / (x.max() - x.min())

    gamma = get_gamma(series, function_target, op)
    arp_series = gamma * x

    return arp_series


def get_gamma(series: np.ndarray, function_target: str, op: str, level: int = 5):

    gamma = (
        level
        if function_target in ["simple", "single"]
        else np.random.uniform(1, level)
    )
    scale = 1 if op == "mult" else np.mean(np.abs(series))
    gamma *= scale
    return gamma
