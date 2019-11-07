from django.db import models

# Create your models here.
class Review(models.Model):
	review = models.TextField(default="Nice Hotel")
	category = models.CharField(max_length=200)
	sentiment = models.BooleanField(default=1)

	def __str__(self):
		return self.review

class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField()
