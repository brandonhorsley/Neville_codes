"""
This code will be to further develop some intuition around what happens as the different
eta values get changed independently. A really sophisticated way to do this could be with 
something like Tkinter and having sliders for each eta value and then the plot gets updated.
If the value gets changed.

Interestingly enough eta1 leads to a horizontal smearing, eta2 leads to a smearing along 
the diagonal, and eta3 leads to a vertical smearing.
"""

#import matplotlib
#matplotlib.use('TkAgg')

from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


def construct_PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat

def construct_BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat

def Calculation(phi1,phi2,eta1,eta2,eta3):
    results=np.empty((len(phi1),len(phi2)))
    a=np.array([1,0])
    a.shape=(2,1)

    b=np.array([1,0])
    b.shape=(1,2)

    for i in range(len(phi1)):
        for j in range(len(phi2)):
            unitary_toy=construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
            results[i][j]=abs(b@unitary_toy@a)**2
    return results

def on_key_event(event):
   print('you pressed %s'%event.key)
   key_press_handler(event, canvas, toolbar)

def mOpen():
   #global t
   global phi1
   global phi2
   global eta1
   global eta2
   global eta3
   global results
   global a
   global canvas_3
   global T1
   global T2
   global T3

   eta1 = DoubleVar()
   eta2 = DoubleVar()
   eta3 = DoubleVar()
   eta1.set(0.5)
   eta2.set(0.5)
   eta3.set(0.5)
   
   slider_1 = Scale(mGui,orient=HORIZONTAL,length = 100,from_=0,to=1, variable=eta1, command=ChangeEta1)
   slider_1.place(x = 700,y=100)
   slider_2 = Scale(mGui,orient=HORIZONTAL,length = 100,from_=0,to=1, variable=eta2, command=ChangeEta2)
   slider_2.place(x = 700,y=200)
   slider_3 = Scale(mGui,orient=HORIZONTAL,length = 100,from_=0,to=1, variable=eta3, command=ChangeEta3)
   slider_3.place(x = 700,y=300)

   l1=Label(text=r"$\eta_1$")
   l1.place(x = 600,y=100)
   l1=Label(text=r"$\eta_2$")
   l1.place(x = 600,y=200)
   l1=Label(text=r"$\eta_3$")
   l1.place(x = 600,y=300)
   
   
   image = Image.open("./Neville_toy.jpg")

   # Resize the image using resize() method
   resize_image = image.resize((500, 300))

   img = ImageTk.PhotoImage(resize_image)

   # create label and add resize image
   label1 = Label(image=img)
   label1.image = img
   label1.place(x=200,y=450)

   T1=Text(height=1,width=5)
   T1.place(x=825,y=100)
   T2=Text(height=1,width=5)
   T2.place(x=825,y=200)
   T3=Text(height=1,width=5)
   T3.place(x=825,y=300)

   phi1=np.linspace(0,2*np.pi,100)
   phi2=phi1
   results=Calculation(phi1,phi2,eta1.get(),eta2.get(),eta3.get())
   
   
   f = Figure(figsize=(5,4),dpi=100)
   a = f.add_subplot(111)
   x=a.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
   #print(eta1.get())
   #print(eta2.get())
   #print(eta3.get())
   f.colorbar(x)
   a.set_xlabel(r"$\phi_1$")
   a.set_ylabel(r"$\phi_2$")
   canvas_3 = FigureCanvasTkAgg(f,master = mGui)
   canvas_3.draw()
   canvas_3.get_tk_widget().place(x=5,y=5)
   toolbar_3 = NavigationToolbar2Tk( canvas_3, mGui )
   toolbar_3.update()
   toolbar_3.place(x=0,y=0)
   canvas_3._tkcanvas.place(x=7,y=7)
   canvas_3.mpl_connect('key_press_event', on_key_event)
   
def ChangeEta1(eta1_name):
    eta1=float(eta1_name)
    #print(r"$\eta_1$ scale is now %.2f" % (eta1))
    T1.delete(1.0,END)
    T1.insert(END,'{:.2f}'.format(eta1))

    results=Calculation(phi1,phi2,eta1,eta2.get(),eta3.get())
    a.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
    canvas_3.draw()
    
def ChangeEta2(eta2_name):
    eta2=float(eta2_name)
    #print(r"$\eta_2$ scale is now %.2f" % (eta2))
    T2.delete(1.0,END)
    T2.insert(END,'{:.2f}'.format(eta2))
    results=Calculation(phi1,phi2,eta1.get(),eta2,eta3.get())
    a.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
    canvas_3.draw()
    
def ChangeEta3(eta3_name):
    eta3=float(eta3_name)
    #print(r"$\eta_3$ scale is now %.2f" % (eta3))
    T3.delete(1.0,END)
    T3.insert(END,'{:.2f}'.format(eta3))
    results=Calculation(phi1,phi2,eta1.get(),eta2.get(),eta3)
    a.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
    canvas_3.draw()

mGui = Tk()                     
mOpen()
mGui.geometry('900x900+300+10') 
mGui.title('Plot')
mGui.mainloop()

"""
Program made to a satisfactory standard.
"""