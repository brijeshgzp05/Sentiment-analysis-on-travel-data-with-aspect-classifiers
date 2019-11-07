from django.urls import path
from .views import review_views,file_views,about_view,home_view
app_name='reviews'

urlpatterns =[
	path('', home_view,name="home"),
	path('/single', review_views,name="single"),
	path('/file', file_views,name="file"),
	#path('/file', review_views,name="review"),
	#path('simple.pngs', file_views,name="file"),
	path('about',about_view, name="about"),

]