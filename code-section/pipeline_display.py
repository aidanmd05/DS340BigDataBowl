"""Plain-text display for audit tables when running outside a notebook."""


def display(*objects):
    for obj in objects:
        print(obj)
