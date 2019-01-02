from config_fasttext1 import Config as Config1
from config_fasttext2 import Config as Config2




def main():
	config = Config1()
	print(config.config_NO)

	config = Config2()
	print(config.config_NO)


if __name__ == "__main__":
	main()
