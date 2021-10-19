#
# urls.py
# group26django
# 
# Created by Jeffrey Wang on 19/10/2021.
# Copyright © 2021 eagersoft.io. All rights reserved.
#

from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    url('^$', views.home),
]
