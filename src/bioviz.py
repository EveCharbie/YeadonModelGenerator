import bioviz
import sys
def main():
    bioviz.Viz(f"{sys.argv[1]}.bioMod").exec()

if __name__ == "__main__":
    main()
