scrapy runspider .\ijra\ijra\spiders\hoghoghi.py -o hoghoghi.json
scrapy runspider .\ijra\ijra\spiders\keyfari.py -o keyfari.json
python Pre_processing.py
python Statistics.py