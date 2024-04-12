from django.http import HttpResponse
from django.shortcuts import render
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.offline as pyo
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from threading import Thread


import matplotlib.pyplot as plt
from KPRMaps.navigator import *

# Create your views here.
def home(request):
    Thread(target=AudioController().audio_input).start()
    # AudioController().audio_input()
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    # Generate the plot

    if(request.method == "POST"):

        start = request.POST.get("start")
        dest = request.POST.get("dest")
        print(start)
        g = Grapher()
        graph = g.create()
        
        x = Navigator(graph)

        try:

            path,cost = x.a_star_algorithm(start,dest)




            plotly_fig = g.load_path(path,cost)

            plot_div = pyo.plot(plotly_fig, output_type='div')

        except:
            return HttpResponse(request, "hi")

        # Pass the HTML string to the template context
        # Render the plot to a PNG image
        # canvas = FigureCanvas(fig)
        # response = HttpResponse(content_type='image/png')
        # canvas.print_png(response)

        return render(request, 'index.html')
    