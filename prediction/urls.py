from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('haberler/', views.news_page, name='news_page'),
    path('performans/', views.lstm_performance_page, name='lstm_performance'),
    path('hisse/<str:ticker>/', views.stock_detail, name='stock_detail'),
]
