from tkinter import *
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
import scipy.stats
from scipy import stats


class Application(Frame):

    # Define settings upon initialization. Here you can specify
    def _init_(self, parent=0):
        Frame._init_(self, parent, bg="thistle3")

        self.top_frameL = Frame(self, bg="thistle3", width=425, height=150)
        self.top_frameR = Frame(self, bg="thistle3", width=425, height=150)

        self.btm_frameR = Frame(self, bg="thistle3", width=425, height=150)
        self.btm_frameR1 = Frame(self, bg="thistle3", width=425, height=150)

        self.top_frameL.pack()
        self.top_frameR.pack()
        self.btm_frameR.pack(anchor='n', fill=BOTH)
        self.btm_frameR1.pack(anchor='n', fill=BOTH)

        w = Label(self.top_frameL, text="Regresion Linear Simple", bg="thistle3", font=("Helvetica", 18, "bold"))
        w.pack(pady=(20, 0))

        self.btn = Button(self.btm_frameR, text='Respuesta!', borderwidth=5, command=self.solutionS,
                          font=("Helvetica", 16))
        self.btn.pack(padx=(280, 10), pady=(30, 0), side=LEFT, anchor='n')

        self.btn1 = Button(self.btm_frameR, text='CE', borderwidth=5, command=self.otra, font=("Helvetica", 16))
        self.btn1.pack(pady=(30, 0), side=LEFT, anchor='n')

        w1 = Label(self.top_frameR, text="Valores de x: ", bg="thistle3", font=("Helvetica", 16, "bold"))
        w1.pack(padx=(0, 10), anchor='nw', pady=(20, 0))

        self.E1 = Entry(self.top_frameR, font=("Helvetica", 14), width=50)
        self.E1.pack(padx=(0, 25), anchor='nw', pady=(0, 20))

        self.w2 = Label(self.top_frameR, text="Valores de y: ", bg="thistle3", font=("Helvetica", 16, "bold"))
        self.w2.pack(padx=(0, 10), anchor='sw')

        self.E2 = Entry(self.top_frameR, font=("Helvetica", 14), width=50)
        self.E2.pack(padx=(0, 25), anchor='sw')

        self.w3 = Label(self.btm_frameR1, text="Regresion Linear Simple: ", bg="thistle3",
                        font=("Helvetica", 16, "bold"))
        self.w3.pack(anchor='n', pady=(25, 0), padx=(20, 0))

        self.listboxF = Listbox(self.btm_frameR1, width=40, height=20, borderwidth=5, font=("Helvetica", 16))
        self.listboxF.pack(padx=(18, 5), pady=(0, 25))

        sys.stdout = self
        self.pack()

        # reference to the parent widget, which is the tk window
        self.parent = parent

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

        # Creation of init_window

    def init_window(self):
        # changing the title of our master widget
        self.master.title("Regresion Linear")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.master)
        self.master.config(menu=menubar)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Regresion Linear Cuadratica", command=self.create_window)
        filemenu.add_command(label="Regresion Linear Multiple", command=self.create_window1)

        filemenu.add_separator()

        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="Programs", menu=filemenu)

    def solutionS(self):
        lista3 = []
        lista4 = []
        lista5 = []
        lista6 = []
        lista7 = []
        a = list(map(float, self.E1.get().split()))
        b = list(map(float, self.E2.get().split()))
        x = np.array(a).reshape((-1, 1))
        y = np.array(b)
        # Multiplicacion de X*X guardada en lista3
        for i in range(len(x)):
            lista3.append(x[i] * x[i])
        lix = np.array(lista3)
        # Multiplicacion de Y*Y guardada en lista4
        for i in range(len(y)):
            lista4.append(y[i] * y[i])
        liy = np.array(lista4)
        # Multiplicacion de X*Y guardada en lista4
        for i in range(len(x)):
            lista5.append(x[i] * y[i])
        lixy = np.array(lista5)
        # Calculo de medias
        xmean = x.mean()
        ymean = y.mean()
        # Calculo de sumas
        sum1 = x.sum()
        sum2 = y.sum()
        sum3 = lix.sum()
        sum4 = liy.sum()
        sum5 = lixy.sum()
        per = float(len(x))
        # Calculo de SSxx y SSxy y ssyy
        ssxx = (sum3 - (math.pow(sum1, 2) / per))
        ssyy = (sum4 - (math.pow(sum2, 2) / per))
        ssxy = (sum5 - (sum1 * sum2 / per))
        # Calculo de b0 y b1
        b1 = (ssxy / ssxx)
        b0 = (ymean - (b1 * xmean))
        # Calculo de ygorrito
        for i in range(len(x)):
            lista6.append((b0 + (b1 * a[i])))
        # Calculo de SSE
        for i in range(len(x)):
            lista7.append(math.pow(b[i] - lista6[i], 2))
        lisse = np.array(lista7)
        # suma de SSE
        sum6 = lisse.sum()
        # Calculo de sb0, sb1, tb0 y tb1
        # Paso 1
        var1 = per - 2
        pas1 = (sum6 / var1)
        # Paso 2
        pas2 = (1 / per + (math.pow(xmean, 2) / ssxx))
        sb0 = math.sqrt(pas1 * pas2)
        sb1 = math.sqrt(sum6 / (var1 * ssxx))
        tb0 = b0 / sb0
        tb1 = b1 / sb1
        # Calculo de S, S^2 y R
        s2 = sum6 / var1
        s = math.sqrt(s2)
        r = (ssxy / (math.sqrt(ssxx * ssyy)))
        model = LinearRegression().fit(x, y)
        self.listboxF.insert(0, "El modelo de regresion es: " + str(round(b0, 3)) + " + " + str(round(b1, 3)) + "x")
        self.listboxF.insert(0, "r: " + " " + str(round(r, 3)))
        self.listboxF.insert(0, "s2: " + " " + str(round(s2, 3)))
        self.listboxF.insert(0, "s: " + " " + str(round(s, 3)))
        self.listboxF.insert(0, "tb1: " + " " + str(round(tb1, 3)))
        self.listboxF.insert(0, "tb0: " + " " + str(round(tb0, 3)))
        self.listboxF.insert(0, "sb1: " + " " + str(round(sb1, 3)))
        self.listboxF.insert(0, "sb0: " + " " + str(round(sb0, 3)))
        self.listboxF.insert(0, "b1: " + " " + str(round(b1, 3)))
        self.listboxF.insert(0, "b0: " + " " + str(round(b0, 3)))
        self.E1.config(state=DISABLED)
        self.E2.config(state=DISABLED)
        self.listboxF.config(state=DISABLED)
        y_pred = model.predict(x)
        plt.scatter(x, y)
        plt.plot(x, y_pred, color='red')
        plt.title("Regresion Linear Simple")
        plt.show()

    def solutionM(self):
        lista1M = []
        lista2M = []
        lista3M = []
        lista4M = []
        lista5M = []
        lista6M = []
        a = list(map(float, self.E3.get().split()))
        b = list(map(float, self.E4.get().split()))
        x = np.array(a)
        y = np.array(b)
        x1 = np.column_stack((x, y))
        # Calculo de x^2
        for i in range(len(y)):
            lista1M.append(x[i] * x[i])
        lixM = np.array(lista1M)
        # Calculo de x^3
        for i in range(len(y)):
            lista2M.append(x[i] * x[i] * x[i])
        lix2M = np.array(lista2M)
        # Calculo de x^4
        for i in range(len(y)):
            lista3M.append(x[i] * x[i] * x[i] * x[i])
        lix3M = np.array(lista3M)
        # Calculo de xy
        for i in range(len(y)):
            lista4M.append(x[i] * y[i])
        lixyM = np.array(lista4M)
        # Calculo de x^2y
        for i in range(len(y)):
            lista5M.append(x[i] * x[i] * y[i])
        lix2yM = np.array(lista5M)
        # Sumas
        sum1M = x.sum()
        sum2M = y.sum()
        sum3M = lixM.sum()
        sum4M = lix2M.sum()
        sum5M = lix3M.sum()
        sum6M = lixyM.sum()
        sum7M = lix2yM.sum()

        # Metodo crammer
        def sarrus(A):
            val = ((A[0][0] * A[1][1] * A[2][2]) +
                   (A[0][1] * A[1][2] * A[2][0]) +
                   (A[0][2] * A[1][0] * A[2][1])) - \
                  ((A[2][0] * A[1][1] * A[0][2]) +
                   (A[2][1] * A[1][2] * A[0][0]) +
                   (A[2][2] * A[1][0] * A[0][1]))
            return val

        sismat = [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]
        res = [0.0, 0.0, 0.0]
        sismat[0][0] = len(y)
        sismat[0][1] = sum1M
        sismat[0][2] = sum3M
        sismat[0][3] = sum2M

        sismat[1][0] = sum1M
        sismat[1][1] = sum3M
        sismat[1][2] = sum4M
        sismat[1][3] = sum6M

        sismat[2][0] = sum3M
        sismat[2][1] = sum4M
        sismat[2][2] = sum5M
        sismat[2][3] = sum7M

        mat_x = [[sismat[0][3], sismat[0][1], sismat[0][2]],
                 [sismat[1][3], sismat[1][1], sismat[1][2]],
                 [sismat[2][3], sismat[2][1], sismat[2][2]]]
        mat_y = [[sismat[0][0], sismat[0][3], sismat[0][2]],
                 [sismat[1][0], sismat[1][3], sismat[1][2]],
                 [sismat[2][0], sismat[2][3], sismat[2][2]]]
        mat_z = [[sismat[0][0], sismat[0][1], sismat[0][3]],
                 [sismat[1][0], sismat[1][1], sismat[1][3]],
                 [sismat[2][0], sismat[2][1], sismat[2][3]]]
        det_mat = sarrus(sismat)
        if det_mat == 0:
            self.listboxM.insert(0, "Determinante de A nulo...")
        else:
            det_matx = sarrus(mat_x)
            det_maty = sarrus(mat_y)
            det_matz = sarrus(mat_z)
            res[0] = det_matx / det_mat
            res[1] = det_maty / det_mat
            res[2] = det_matz / det_mat
            self.listboxM.insert(0, "P => " + " " + str(res))
        A = res[0]
        b1 = res[1]
        b2 = res[2]
        ##Calculo de ygorrito
        for i in range(len(y)):
            lista6M.append((b2 * lixM[i]) + (b1 * x[i]) + A)
        liygorr = np.array(lista6M)
        pf = PolynomialFeatures(degree=2)
        X = pf.fit_transform(x.reshape(-1, 1))
        regresion_lineal = LinearRegression()
        regresion_lineal.fit(X, y)
        r2 = regresion_lineal.score(X, y)
        self.listboxM.insert(0, "El modelo de regresion es: " + str(round(b2, 3)) + "x^2" + " + " + str(
            round(b1, 3)) + "x" + " + " + str(
            round(A, 3)))
        self.listboxM.insert(0, "r2: " + " " + str(round(r2, 3)))
        self.listboxM.insert(0, "liygorr: " + " " + str(liygorr))
        matri = []
        for i in range(len(b)):
            matri.append([b[i], liygorr[i], abs(round(b[i] - liygorr[i], 3))])
        data = DataFrame(matri, columns=['y', 'ygorr', 'diff'])
        self.listboxM.insert(0, data)
        poly_reg = PolynomialFeatures(degree=2)
        X_poly = poly_reg.fit_transform(x.reshape(-1, 1))
        lin_reg2 = LinearRegression()
        lin_reg2.fit(X_poly, y.reshape(-1, 1))
        y_pred = lin_reg2.predict(X_poly)
        plt.scatter(x, y, color='red')
        plt.plot(x, y_pred)
        plt.title("Regresion Linear Cuadratica")
        plt.show()

    def solutionX(self):
        lista2X = []
        lista3X = []
        lista4X = []
        lista5X = []
        lista6X = []
        lista7X = []
        lista8X = []
        lista9X = []
        lista10X = []
        lista11X = []
        a = list(map(float, self.E5.get().split()))
        b = list(map(float, self.E6.get().split()))
        c = list(map(float, self.E7.get().split()))
        x = np.array(a)  # x2
        y = np.array(b)  # x1
        z = np.array(c)  # y
        k = 2
        # Multiplicacion de X1*X1 guardada en lista3
        for i in range(len(x)):
            lista2X.append(x[i] * x[i])
        lix1X = np.array(lista2X)
        # Multiplicacion de X2*X2 guardada en lista3
        for i in range(len(x)):
            lista3X.append(y[i] * y[i])
        lix2X = np.array(lista3X)
        # Multiplicacion de X1*X2 guardada en lista3
        for i in range(len(x)):
            lista4X.append(a[i] * y[i])
        lix1x2X = np.array(lista4X)
        # Multiplicacion de X1*Y guardada en lista3
        for i in range(len(x)):
            lista5X.append(a[i] * z[i])
        lix1yX = np.array(lista5X)
        # Multiplicacion de X2*Y guardada en lista3
        for i in range(len(x)):
            lista6X.append(y[i] * z[i])
        lix2yX = np.array(lista6X)
        # Multiplicacion de Y*Y guardada en lista3
        for i in range(len(x)):
            lista7X.append(z[i] * z[i])
        liyyX = np.array(lista7X)
        # Calculo de sumas
        sum1X = x.sum()
        sum2X = y.sum()
        sum3X = z.sum()
        sum4X = lix1X.sum()
        sum5X = lix2X.sum()
        sum6X = lix1x2X.sum()
        sum7X = lix1yX.sum()
        sum8X = lix2yX.sum()
        sum9X = liyyX.sum()

        ##Regla de cramer
        def sarrus(A):
            val = ((A[0][0] * A[1][1] * A[2][2]) +
                   (A[0][1] * A[1][2] * A[2][0]) +
                   (A[0][2] * A[1][0] * A[2][1])) - \
                  ((A[2][0] * A[1][1] * A[0][2]) +
                   (A[2][1] * A[1][2] * A[0][0]) +
                   (A[2][2] * A[1][0] * A[0][1]))
            return val

        sismat = [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]
        res = [0.0, 0.0, 0.0]
        sismat[0][0] = len(x)
        sismat[0][1] = sum1X
        sismat[0][2] = sum2X
        sismat[0][3] = sum3X
        sismat[1][0] = sum1X
        sismat[1][1] = sum4X
        sismat[1][2] = sum6X
        sismat[1][3] = sum7X
        sismat[2][0] = sum2X
        sismat[2][1] = sum6X
        sismat[2][2] = sum5X
        sismat[2][3] = sum8X
        mat_x = [[sismat[0][3], sismat[0][1], sismat[0][2]],
                 [sismat[1][3], sismat[1][1], sismat[1][2]],
                 [sismat[2][3], sismat[2][1], sismat[2][2]]]
        mat_y = [[sismat[0][0], sismat[0][3], sismat[0][2]],
                 [sismat[1][0], sismat[1][3], sismat[1][2]],
                 [sismat[2][0], sismat[2][3], sismat[2][2]]]
        mat_z = [[sismat[0][0], sismat[0][1], sismat[0][3]],
                 [sismat[1][0], sismat[1][1], sismat[1][3]],
                 [sismat[2][0], sismat[2][1], sismat[2][3]]]
        det_mat = sarrus(sismat)
        if det_mat == 0:
            self.listboxX.insert(0, "Determinante de A nulo...")
        else:
            det_matx = sarrus(mat_x)
            det_maty = sarrus(mat_y)
            det_matz = sarrus(mat_z)
            res[0] = det_matx / det_mat
            res[1] = det_maty / det_mat
            res[2] = det_matz / det_mat
            self.listboxX.insert(0, "P => " + str(res))
        A = res[0]
        b1 = res[1]
        b2 = res[2]
        self.listboxX.insert(0, "El modelo de regresion es: " + str(round(A, 3)) + " + " + str(round(b1, 3)) + "x1",
                             " + " + str(round(b2, 3)) + "x2")
        # Calculo de ygorrito
        for i in range(len(x)):
            lista8X.append((A + (b1 * x[i]) + (b2 * y[i])))
        liygorrX = np.array(lista8X)
        sum10X = liygorrX.sum()
        # Calculo de SSE
        for i in range(len(x)):
            lista9X.append(math.pow((z[i] - liygorrX[i]), 2))
        liysseX = np.array(lista9X)
        sum11X = liysseX.sum()
        prom = z.mean()
        # Calculo de variacion total
        for i in range(len(x)):
            lista10X.append(math.pow(z[i] - prom, 2))
        livartX = np.array(lista10X)
        # Calculo de variacion Explicada
        for i in range(len(x)):
            lista11X.append(math.pow(lista8X[i] - prom, 2))
        livarexX = np.array(lista11X)
        sum12X = livarexX.sum()
        # Calculo de variacion inexplicada
        livarinX = liysseX
        sum13X = sum11X
        # Calculo de S^2 y S
        s2 = (sum11X / (len(z) - (k + 1)))
        s = math.sqrt(s2)
        # Calculo de Variacion total, r^2, r
        vartotal = sum12X + sum11X
        r2 = sum12X / vartotal
        r = math.sqrt(r2)
        # R ajustada
        var1 = k / (len(x) - 1)
        var2 = ((len(x) - 1) / (len(x) - (k + 1)))
        rajustado = (r2 - var1) * var2
        # Calculo de valores f
        f1 = (sum12X / k)
        f2 = (sum13X / (len(x) - (k + 1)))
        fmodelo = (f1 / f2)

        q = scipy.stats.f.ppf(q=1 - 0.05, dfn=k, dfd=(len(x) - (k + 1)))
        falpha = round(q, 3)

        self.listboxX.insert(0, str(round((stats.t.ppf(1 - 0.025, (len(x) - (k + 1)))), 3)))
        self.listboxX.insert(0, "s2: " + " " + str(round(s2, 3)))
        self.listboxX.insert(0, "s: " + " " + str(round(s, 3)))
        self.listboxX.insert(0, "r2: " + " " + str(round(r2, 3)))
        self.listboxX.insert(0, "r: " + " " + str(round(r, 3)))
        self.listboxX.insert(0, "rajustado: " + str(round(rajustado, 3)))
        self.listboxX.insert(0, "fmodelo: " + str(round(fmodelo, 3)))
        self.listboxX.insert(0, "falpha: " + str(round(falpha, 3)))
        self.listboxX.insert(0, "fmodelo: " + str(round(fmodelo, 3)))

    def create_window(self):
        newWindow = Toplevel(self)
        newWindow.title("Regresion Linear Cuadratico")
        newWindow.config(bg='thistle3')
        newWindow.geometry('{}x{}'.format(775, 675))
        newWindow.resizable(False, False)
        self.top = Frame(newWindow, bg='thistle3')
        self.top1 = Frame(newWindow, bg='thistle3')
        self.mid = Frame(newWindow, bg='thistle3')
        self.mid1 = Frame(newWindow, bg='thistle3')

        self.top.pack()
        self.top1.pack()
        self.mid.pack(anchor='n', fill=BOTH)
        self.mid1.pack(anchor='n', fill=BOTH)

        self.Label4 = Label(self.top, text="Regresion Linear Cuadratica", bg="thistle3", font=("Helvetica", 18, "bold"))
        self.Label4.pack(pady=(20, 0))

        self.btnM = Button(self.mid, text='Respuesta!', borderwidth=5, command=self.solutionM, font=("Helvetica", 16))
        self.btnM.pack(padx=(280, 10), pady=(30, 0), side=LEFT, anchor='n')

        self.btnM1 = Button(self.mid, text='CE', borderwidth=5, command=self.otraM, font=("Helvetica", 16))
        self.btnM1.pack(pady=(30, 0), side=LEFT, anchor='n')

        self.Label5 = Label(self.top1, text="Valores de x: ", bg="thistle3", font=("Helvetica", 16, "bold"))
        self.Label5.pack(padx=(0, 10), anchor='nw', pady=(20, 0))

        self.E3 = Entry(self.top1, font=("Helvetica", 14), width=50)
        self.E3.pack(padx=(0, 25), anchor='nw', pady=(0, 20))

        self.Label6 = Label(self.top1, text="Valores de y: ", bg="thistle3", font=("Helvetica", 16, "bold"))
        self.Label6.pack(padx=(0, 10), anchor='sw')

        self.E4 = Entry(self.top1, font=("Helvetica", 14), width=50)
        self.E4.pack(padx=(0, 25), anchor='sw')

        self.Label6 = Label(self.mid1, text="Regresion Linear Cuadratica: ", bg="thistle3",
                            font=("Helvetica", 16, "bold"))
        self.Label6.pack(anchor='n', pady=(25, 0), padx=(20, 0))

        self.listboxM = Listbox(self.mid1, width=40, height=20, borderwidth=5, font=("Helvetica", 16))
        self.listboxM.pack(padx=(18, 5), pady=(0, 25))

    def create_window1(self):
        newWindow1 = Toplevel(self)
        newWindow1.geometry('{}x{}'.format(750, 750))
        newWindow1.resizable(False, False)
        newWindow1.config(bg="thistle3")
        newWindow1.title('Regresion Linear Multiple')
        self.top_frame1 = Frame(newWindow1, bg='thistle3')
        self.top_frame2 = Frame(newWindow1, bg='thistle3')
        self.btm_frame1 = Frame(newWindow1, bg='thistle3')
        self.btm_frame2 = Frame(newWindow1, bg='thistle3')

        self.top_frame1.pack()
        self.top_frame2.pack()
        self.btm_frame1.pack(anchor='n', fill=BOTH)
        self.btm_frame2.pack(anchor='n', fill=BOTH)

        self.Label7 = Label(self.top_frame1, text="Regresion Linear Multiple", bg="thistle3", font=("Helvetica", 18, "bold"))
        self.Label7.pack(pady=(20, 0))

        self.btnX = Button(self.btm_frame1, text='Respuesta!', borderwidth=5, command=self.solutionX, font=("Helvetica", 16))
        self.btnX.pack(padx=(280, 10), pady=(30, 0), side=LEFT, anchor='n')

        self.btnX1 = Button(self.btm_frame1, text='CE', borderwidth=5, command=self.otraX, font=("Helvetica", 16))
        self.btnX1.pack(pady=(30, 0), side=LEFT, anchor='n')

        self.Label8 = Label(self.top_frame2, text="Valores de x: ", bg="thistle3", font=("Helvetica", 16, "bold"))
        self.Label8.pack(padx=(0, 10), anchor='nw', pady=(20, 0))

        self.E5 = Entry(self.top_frame2, font=("Helvetica", 14), width=50)
        self.E5.pack(padx=(0, 25), anchor='nw', pady=(0, 20))

        self.Label9 = Label(self.top_frame2, text="Valores de x1: ", bg="thistle3", font=("Helvetica", 16, "bold"))
        self.Label9.pack(padx=(0, 10), anchor='nw', pady=(20, 0))

        self.E6 = Entry(self.top_frame2, font=("Helvetica", 14), width=50)
        self.E6.pack(padx=(0, 25), anchor='nw', pady=(0, 20))

        self.Label10 = Label(self.top_frame2, text="Valores de y: ", bg="thistle3", font=("Helvetica", 16, "bold"))
        self.Label10.pack(padx=(0, 10), anchor='sw')

        self.E7 = Entry(self.top_frame2, font=("Helvetica", 14), width=50)
        self.E7.pack(padx=(0, 25), anchor='sw')

        self.Label11 = Label(self.btm_frame2, text="Regresion Linear Multiple: ", bg="thistle3",
                             font=("Helvetica", 16, "bold"))
        self.Label11.pack(anchor='n', pady=(25, 0), padx=(20, 0))

        self.listboxX = Listbox(self.btm_frame2, width=40, height=20, borderwidth=5, font=("Helvetica", 16))
        self.listboxX.pack(padx=(18, 5), pady=(0, 25))

    def otra(self):
        self.E1.config(state=NORMAL)
        self.E2.config(state=NORMAL)
        self.listboxF.config(state=NORMAL)
        plt.close()
        self.E1.delete(0, 'end')
        self.E2.delete(0, 'end')
        self.listboxF.delete(0, 'end')

    def otraM(self):
        self.E3.config(state=NORMAL)
        self.E4.config(state=NORMAL)
        self.listboxM.config(state=NORMAL)
        plt.close()
        self.E3.delete(0, 'end')
        self.E4.delete(0, 'end')
        self.listboxM.delete(0, 'end')

    def otraX(self):
        self.E5.config(state=NORMAL)
        self.E6.config(state=NORMAL)
        self.E7.config(state=NORMAL)
        self.listboxX.config(state=NORMAL)
        plt.close()
        self.E5.delete(0, 'end')
        self.E6.delete(0, 'end')
        self.E7.delete(0, 'end')
        self.listboxX.delete(0, 'end')

    def flush(self):
        pass


root = Tk()
root.geometry("725x650")
root.resizable(False, False)
# creation of an instance
app = Application()
# mainloop
mainloop()