import typing as tp

from copy import deepcopy

import etna

from etna.pipeline import Pipeline


def _add_attrs(
        pipe: Pipeline,
        attrs: tp.List[tp.Any],
        attr_name: str
    ) -> tp.List[Pipeline]:
    """auxiliary function to stack pipelines 

    Args:
        pipe (Pipeline): pipeline to stack from
        attrs (tp.List[tp.Any]): attributes to set
        attr_name (str): name of pipeline attribute to set
    Returns:
        tp.List[Pipeline]: pipelines where each pipeline is a
          full copy of "pipe", except "attr_name" attribute.
          len(result) == len(atrrs)
    """
    result = []
    for a in attrs:
        p = deepcopy(pipe)
        setattr(p, attr_name, a)
        result.append(p)

    return result


def add_horizons(
        pipe: Pipeline,
        horizons: tp.List[int]
    ) -> tp.List[Pipeline]:
    """stack many horizons to test

    Args:
        pipe (Pipeline):
        horizons (tp.List[int]):

    Returns:
        tp.List[Pipeline]:
    """
    return _add_attrs(pipe, horizons, 'horizon')


def add_transforms(
        pipe: Pipeline,
        transforms: tp.List[etna.transforms.base.ReversibleTransform]
) -> tp.List[Pipeline]:
    """ stack many transforms to test

    Args:
        pipe (Pipeline): _description_
        transforms (tp.List): _description_

    Returns:
        tp.List[Pipeline]: _description_
    """
    return _add_attrs(pipe, transforms, 'transforms')


def add_models(
        pipe: Pipeline,
        kw_args: tp.List[dict],
        ModelClass: tp.Type
) -> tp.List[Pipeline]:
    """Similarly to add_horizons, and add_transforms, but
      changes pipeline's model attribute

    Args:
        pipe (Pipeline):
        kw_args (tp.List[dict]):
        ModelClass (tp.Type):

    Returns:
        tp.List[Pipeline]:
    """
    res = []
    for ka in kw_args:
        p = deepcopy(pipe)
        p.model = ModelClass(**ka)
        res.append(p)

    return res


def get_pipelines(
        horizons: tp.List[int],
        transforms: tp.List[etna.transforms.base.ReversibleTransform],
        kw_args: tp.List[dict],
        ModelClass: tp.Type,
        pipe: tp.Optional[Pipeline] = None
) -> tp.List[Pipeline]:
    """stack horizons, transforms and models
    Replaces 3 for loops logic

    Args:
        horizons (tp.List[int]):
        transforms (tp.List[etna.transforms.base.ReversibleTransform]):
        kw_args (tp.List[dict]):
        ModelClass (tp.Type):
        pipe (tp.Optional[Pipeline], optional):

    Returns:
        tp.List[Pipeline]:
    """
    if pipe is None:
        pipe = Pipeline(
            model = ModelClass(),
            transforms=transforms,
            horizon=horizons[0]
        )
    assert horizons

    res = add_horizons(pipe, horizons)

    ans = []
    for p in res:
        pipes = add_transforms(p, transforms)
        ans.extend(pipes)
    if ans:
        res = ans

    ans = [] 
    for p in res:
        pipes = add_models(p, kw_args, ModelClass)
        ans.extend(pipes)
    if ans:
        res = ans

    return res
