from django.shortcuts import render,HttpResponseRedirect,redirect,HttpResponse
#from .models import Review
from .forms import ReviewForm
from django.urls import reverse
from django.contrib import messages

from .models import Document
from .forms import DocumentForm
import csv
import pandas

import re
import pickle
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.preprocessing.sequence import pad_sequences

import os
from django.conf  import settings


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt 
import numpy as np

import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(settings.BASE_DIR, 'model_1_lac.bin'), binary=True, limit=100000)


loaded_model = keras.models.load_model(os.path.join(settings.BASE_DIR, 'model_16_h_rev.h5'))
with open(os.path.join(settings.BASE_DIR, 'tokenizer.pickle'), 'rb') as handle:
	loaded_tokenizer = pickle.load(handle)


def home_view(request):
	return render(request, 'home.html', {})

def review_views(request):
	form = ReviewForm(request.POST or None)
	if form.is_valid():
		x = request.POST['review']
		x = preprocess_text(x)

		# sentiment analysis
		pre1 = []
		pre=x
		pre1.append(x)
		print(pre1)
		val = sentiment_function(pre1)
		# aspect classifier
		final_res = aspect_extractor(pre) 
		form = ReviewForm()
		context={
		'forms':form,
		'result': final_res,
		'rev':x,
		'sentiment':val,
		}
		return render(request, 'single_result.html' ,context)
	
	context ={
		'forms':form,

	}

	return render(request,'review.html',context)	


def file_views(request):

	if request.method == 'POST':
		form = DocumentForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			file = request.FILES['document']
			#name = request.POST['description']
			add_ = file.name
			fields = [] 
			rows = []
			cols = [x.replace(' ', '_') for x in add_.split()]
			new_add = ''
			for i in cols:
				if '.' in i:
					new_add+=i 
				else:
					new_add+=i+'_'	

			print(new_add)

			df = pandas.read_csv(new_add)

			pos = [0]*7
			neg = [0]*7

			for i in range(len(df)):
				x = df.loc[i,'Review_text']
				x = preprocess_text(x)

				# sentiment analysis
				pre1 = []
				pre=x
				pre1.append(x)
				print(pre1)
				val = sentiment_function(pre1)
				# aspect classifier
				final_res = aspect_extractor(pre)

				if final_res == 'Cleanliness':
					if val==1:
						pos[0] += 1
					else:
						neg[0] += 1

				elif final_res == 'Staff':
					if val==1:
						pos[1] += 1
					else:
						neg[1] += 1

				elif final_res == 'Internet':
					if val==1:
						pos[2] += 1
					else:
						neg[2] += 1


				elif final_res == 'Food':
					if val==1:
						pos[3] += 1
					else:
						neg[3] += 1

				elif final_res == 'Location':
					if val==1:
						pos[4] += 1
					else:
						neg[4] += 1

				elif final_res == 'VfM':
					if val==1:
						pos[5] += 1
					else:
						neg[5] += 1

				else:
					if val==1:
						pos[6] += 1
					else:
						neg[6] += 1


			
			n = 7
			ind = np.arange(n)    # the x locations for the groups
			width = 0.35       # the width of the bars: can also be len(x) sequence

			
			fig, ax = plt.subplots()
			ax.bar(ind, pos, width, color='green')
			ax.bar(ind,neg,width,bottom=pos,color='red')
			ax.set(ylabel ='Level of measurement', title = 'Analysis of various fields')
			ax.set_xticks(ind)
			ax.set_xticklabels(['Cleanliness', 'Staff', 'Internet', 'Food', 
				'Location','VfM','C&F'])
			ax.set_xticks(ind,(	))
			ax.grid()




			canvas=FigureCanvas(fig)
			response = HttpResponse(content_type='image/jpg')
			canvas.print_jpg(response)
			return response
			 	

			return redirect(reverse('review:review'))
	else:
		form = DocumentForm()
		return render(request, 'file.html', {'form': form})


	 

	



# def review_history(request):
# 	review = Review.objects.all()
# 	print(review)

# 	reviewlist = []
# 	for i in review:
# 		rev = i
# 		reviewlist.append(rev)
# 	print(reviewlist)	
# 	context={
# 		'rev_obj':review,
# 		'reviewlist':reviewlist,
		
# 	}
# 	return render(request,"result.html",context)


def about_view(request):
	return render(request,"about.html",{})








#--------------------------------------------------------------------------



def preprocess_text(x):
        x = re.sub(r'https:\/\/[a-zA-Z]*\.com',' ',x)
        x = re.sub(r'\W+',' ',x)
        x = re.sub(r'\d+',' ',x)
        x = re.sub(r'\s[a-zA-Z]\s',' ',x)
        x = re.sub(r'^\s+','',x)
        x = re.sub(r'\s+$','',x)
        x = re.sub(r'\s+',' ',x)
        x = x.lower()
        return x


def weight_calculator(aspect, text):
    weight = 0
    words = word_tokenize(text)
    removed_stopwords_list = [word for word in words if word not in stopwords.words('english')]
    lemmatized_words_list = [lemmatizer.lemmatize(word) for word in removed_stopwords_list]
    for w in lemmatized_words_list:
        try:
            weight += model.wv.similarity(aspect,w)
        except:
            pass
    return weight


def aspect_extractor(text):
    text = preprocess_text(text)
    dic_ = dict()
    dic_['Cleanliness'] = weight_calculator('clean',text)
    dic_['Staff'] = weight_calculator('staff',text)
    dic_['Internet'] = weight_calculator('internet',text)
    dic_['Food'] = weight_calculator('food',text)
    dic_['Location'] = weight_calculator('location',text)
    dic_['VfM'] = weight_calculator('price',text)
    com = weight_calculator('comfort',text)
    fac = weight_calculator('facility',text)
    dic_['C&F'] = max(com,fac)
    #sns.barplot(x = dic_.keys(),y=dic_.values())
    max_weight = -1
    aspect = 'C&F'
    for key,value in dic_.items():
        if value>max_weight:
            max_weight = value
            aspect = key
    return aspect


def sentiment_function(pre1):
	max_length=324
	trunc_type = 'post'
	pre_sequences = loaded_tokenizer.texts_to_sequences(pre1)
	print(pre_sequences)
	pre_padded = pad_sequences(pre_sequences,maxlen=max_length, truncating=trunc_type)
	prediction = loaded_model.predict(pre_padded)
	prediction = prediction[0][0]
	if prediction>=0.55:
		val = 1
		print('positive, confidence : ', str(prediction))
	else:
		val = 0
		print('negative, confidence : ', str(prediction))
	return val 