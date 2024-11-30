from django.contrib import admin
from django.urls import path  # pathをインポート
from django.conf import settings
from django.conf.urls.static import static  # staticをインポート
from . import views

urlpatterns = [
    # 他のパス設定
    path('', views.index, name='index'),  # ルート URL に対するビューを追加
    path('upload/', views.upload_video, name='upload_video'),
    path('result/', views.result, name='result'),
    # その他のURLパターン
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
