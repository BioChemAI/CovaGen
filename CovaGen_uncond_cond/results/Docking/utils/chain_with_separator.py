from typing import Iterable, Any, Generator

def chain_with_separator(iterables: Iterable, separator: Any) -> Generator:
    for it in iterables:
        for element in it:
            yield element
        yield separator

def chain_with_separator_and_segment(iterables: Iterable, separator: Any) -> Generator:
    for segment, it in enumerate(iterables):
        for element in it:
            yield segment, element
        yield segment, separator
