echo "Making no robots..."
python ./scripts/data_scripts/make_no_robots.py

echo "Making twitter..."
git clone https://github.com/IBM/twitter-customer-care-document-prediction.git
python ./scripts/data_scripts/make_twitter.py
