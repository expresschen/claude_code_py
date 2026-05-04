"""Async generator utilities.

These mirror the TypeScript async generator patterns used in claude-code.
"""

from __future__ import annotations

import asyncio
from typing import (
    AsyncGenerator,
    AsyncIterable,
    Callable,
    TypeVar,
    Awaitable,
)

T = TypeVar("T")
R = TypeVar("R")


async def all(
    generators: list[Callable[[], AsyncGenerator[T, None]]],
    concurrency: int = 10,
) -> AsyncGenerator[T, None]:
    """Run multiple async generators in parallel with concurrency limit.

    This mirrors the TypeScript `all()` function for generator composition.

    Args:
        generators: List of async generator factory functions
        concurrency: Maximum number of concurrent generators

    Yields:
        Items from all generators in completion order
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_semaphore(
        gen_factory: Callable[[], AsyncGenerator[T, None]]
    ) -> AsyncGenerator[T, None]:
        async with semaphore:
            async for item in gen_factory():
                yield item

    # Create tasks for all generators
    tasks: list[asyncio.Task[None]] = []
    queue: asyncio.Queue[T | None] = asyncio.Queue()
    active_count = len(generators)

    async def consume_generator(gen_factory: Callable[[], AsyncGenerator[T, None]]) -> None:
        try:
            async for item in run_with_semaphore(gen_factory):
                await queue.put(item)
        finally:
            nonlocal active_count
            active_count -= 1
            if active_count == 0:
                await queue.put(None)  # Signal completion

    # Start all tasks
    for gen_factory in generators:
        task = asyncio.create_task(consume_generator(gen_factory))
        tasks.append(task)

    # Yield items as they become available
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)


async def merge_generators(
    *generators: AsyncGenerator[T, None],
) -> AsyncGenerator[T, None]:
    """Merge multiple async generators into one.

    Args:
        generators: Async generators to merge

    Yields:
        Items from all generators in completion order
    """
    queue: asyncio.Queue[T | None] = asyncio.Queue()
    active_count = len(generators)

    async def consume(gen: AsyncGenerator[T, None]) -> None:
        try:
            async for item in gen:
                await queue.put(item)
        finally:
            nonlocal active_count
            active_count -= 1
            if active_count == 0:
                await queue.put(None)

    # Start consumers
    tasks = [asyncio.create_task(consume(gen)) for gen in generators]

    # Yield items
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

    # Wait for completion
    await asyncio.gather(*tasks, return_exceptions=True)


async def async_generator_to_list(gen: AsyncGenerator[T, None]) -> list[T]:
    """Collect all items from an async generator into a list.

    Args:
        gen: Async generator

    Returns:
        List of all yielded items
    """
    result: list[T] = []
    async for item in gen:
        result.append(item)
    return result


async def map_generator(
    gen: AsyncGenerator[T, None],
    mapper: Callable[[T], Awaitable[R] | R],
) -> AsyncGenerator[R, None]:
    """Map items from an async generator.

    Args:
        gen: Source generator
        mapper: Function to apply to each item

    Yields:
        Mapped items
    """
    async for item in gen:
        result = mapper(item)
        if asyncio.iscoroutine(result):
            yield await result
        else:
            yield result


async def filter_generator(
    gen: AsyncGenerator[T, None],
    predicate: Callable[[T], Awaitable[bool] | bool],
) -> AsyncGenerator[T, None]:
    """Filter items from an async generator.

    Args:
        gen: Source generator
        predicate: Filter function

    Yields:
        Items where predicate returns True
    """
    async for item in gen:
        should_include = predicate(item)
        if asyncio.iscoroutine(should_include):
            should_include = await should_include
        if should_include:
            yield item


async def take_generator(
    gen: AsyncGenerator[T, None],
    n: int,
) -> AsyncGenerator[T, None]:
    """Take at most n items from an async generator.

    Args:
        gen: Source generator
        n: Maximum number of items

    Yields:
        At most n items
    """
    count = 0
    async for item in gen:
        if count >= n:
            break
        yield item
        count += 1