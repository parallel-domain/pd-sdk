import random
from typing import Callable, Generator, TypeVar

A = TypeVar("A")
B = TypeVar("B")


def nested_generator_round_robin_draw(
    source_generator: Generator[A, None, None],
    nested_generator_factory: Callable[[A], Generator[B, None, None]],
    endless_loop: bool,
    random_seed: int,
) -> Generator[B, None, None]:
    nested_gens = []
    state = random.Random(random_seed)
    # collect generators of all source objects. Yield one item from each to avoid
    # delay caused by iterating over source_generator
    for source_obj in source_generator:
        nested_gen = nested_generator_factory(source_obj)
        item = next(nested_gen, None)
        if item is not None:
            nested_gens.append((source_obj, nested_gen))
            yield item

    # while there are nested_gens that still have items to yield
    # yield one item of a source_objs gen and then move on to the next source_objs gen
    while len(nested_gens) > 0:
        idx = state.choice(range(len(nested_gens)))
        source_obj, nested_gen = nested_gens.pop(idx)
        item = next(nested_gen, None)
        if item is not None:
            nested_gens.append((source_obj, nested_gen))
            yield item
        elif endless_loop:
            nested_gen = nested_generator_factory(source_obj)
            nested_gens.append((source_obj, nested_gen))
