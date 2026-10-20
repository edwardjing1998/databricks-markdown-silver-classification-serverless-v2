"""Book-ID based source isolation for independent classification jobs."""


def parse_book_ids(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(x.strip() for x in (value or "").split(",") if x.strip()))


def source_allowed(book_id: str, *, include: str = "", exclude: str = "") -> bool:
    includes = parse_book_ids(include)
    excludes = parse_book_ids(exclude)
    return (not includes or book_id in includes) and book_id not in excludes
