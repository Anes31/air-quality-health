from src.parse_air_quality import main as parse_main


def main():
    print("=== Running hourly ETL (parse_air_quality) ===")
    parse_main()
    print("=== Hourly ETL finished ===")


if __name__ == "__main__":
    main()
