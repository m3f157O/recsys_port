import GowallaReader
import AmazonReader
import YelpReader
def main():
    pre_splitted_path = "DatasetPublic/data/Gowalla/"
    reader = YelpReader.YelpReader(pre_splitted_path)
    reader = AmazonReader.AmazonReader(pre_splitted_path)
    reader = GowallaReader.GowallaReader(pre_splitted_path)


if __name__ == '__main__':
    main()