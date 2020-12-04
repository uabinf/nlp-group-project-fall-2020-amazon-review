"""
You can add the CLI you were talking about to load a trained model and run predictions. You can also try optimizing the results. One option is to use DistillBert instead of Bert to see if that will improve the time needed for training. Furthermore, I left a TODO to handle reading the dataset from file instead of holding the whole thing in memory. I assume there will be issues with some of the larger datasets (like Electronics).

reading dataset from file: just save it then read
write output to file

the jupyter notebook saves the weights of the model. the model can be loaded from memory

if you wanted to create the cli, you'd have to import all the stuff present in the current notebook, then load the weights from disk, et al and have *that* execute from the file

add stuff to the slideshow concerning anything that I add.
"""
import torch
MODEL_SAVE_PATH = '/data/user/jprob/bert2gpt'
MODEL_TEMP_PATH = '/data/user/jprob/temp'
def call(req_mod):
	if req_mod == None:
		model = EncoderDecoderModel.from_pretrained(MODEL_SAVE_PATH)
		product = list(product_dataset.values())[0]
		product_category = product['category']
		product_title = product['title']
		product_description = product['description']
		product_combined = generate_input_sequence(5, product_category, product_title, product_description)
		input_ids, _ = preprocess_encoder_input(product_combined)
		input_ids = torch.tensor(input_ids).unsqueeze(0)
		output_ids = model.generate(
			input_ids,
			decoder_start_token_id=model.config.decoder.pad_token_id,
			temperature=1.3,
			top_k=9,
			top_p=0.9,
			repetition_penalty=1.4
		)
		with open("generate.txt", "w") as f:
			f.write(product)
			f.write(output_ids)
	else:
		raise NotImplementedError

data_metadata = [
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/AMAZON_FASHION.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_AMAZON_FASHION.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_All_Beauty.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Appliances.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Appliances.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Arts_Crafts_and_Sewing.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Arts_Crafts_and_Sewing.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Automotive.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Automotive.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Books.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Books.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/CDs_and_Vinyl.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_CDs_and_Vinyl.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Cell_Phones_and_Accessories.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Cell_Phones_and_Accessories.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Clothing_Shoes_and_Jewelry.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Digital_Music.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Digital_Music.json.gz"),
	('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Electronics.json.gz', 'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Electronics.json.gz'),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Gift_Cards.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Gift_Cards.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Grocery_and_Gourmet_Food.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Grocery_and_Gourmet_Food.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Home_and_Kitchen.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Home_and_Kitchen.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Industrial_and_Scientific.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Industrial_and_Scientific.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Kindle_Store.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Kindle_Store.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Luxury_Beauty.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Luxury_Beauty.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Magazine_Subscriptions.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Magazine_Subscriptions.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Movies_and_TV.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Musical_Instruments.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Musical_Instruments.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Office_Products.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Office_Products.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Patio_Lawn_and_Garden.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Patio_Lawn_and_Garden.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Pet_Supplies.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Pet_Supplies.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Prime_Pantry.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Prime_Pantry.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Software.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Software.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Sports_and_Outdoors.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Sports_and_Outdoors.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Tools_and_Home_Improvement.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Tools_and_Home_Improvement.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Toys_and_Games.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Toys_and_Games.json.gz"),
	("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz", "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Video_Games.json.gz")
]

def get():
	return input("1: Use current trained model\n2: Train on different data\n")
def getmodel():
	return input("Pick one: ")

if __name__ == "__main__":
	ans = get()
	while ans not in ['1', '2']:
		ans = get()
	if ans == '1':
		call(None)
	elif ans == '2':
		for count, i in enumerate(data_metadata):
			print(f"\t{count + 1}: {i[0][53:].rstrip('.json.gz')}")
		ans = getmodel()
		while ans not in [str(i) for i in range(1, len(data_metadata) + 1)]:
			ans = getmodel()
		call(data_metadata[int(ans) - 1])
