"""Compatibility entry point: use summary.py --flops for an explicit FLOP count."""

from summary import main


if __name__ == "__main__":
    import sys
    if "--flops" not in sys.argv:
        sys.argv.append("--flops")
    main()
