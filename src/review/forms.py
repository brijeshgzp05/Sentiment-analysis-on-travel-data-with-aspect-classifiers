from django import forms
from .models import Review, Document


class ReviewForm(forms.ModelForm):
	review 	= forms.CharField(label="" ,widget=forms.TextInput(attrs={'class':'form-control','placeholder':'','autocomplete':'off'}))
	# def clean_review(self,*args,**kwargs):
	# 	review = self.cleaned_data.get("review")
		#print(review)
	class Meta:
		model = Review
		fields = ['review']


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document',)