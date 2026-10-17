"""Run inside the committed H087 image on one GPU; see --help."""

from open_instruct.miles.rewards import h087_qualification

if __name__ == "__main__":
    h087_qualification.main()
