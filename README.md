#  Age-Gender Detector

Una empresa que se dedica a tratamiento de imágenes quiere exponer en su página web una aplicación que le permita a los usuarios a partir de una foto, detectar su género y su edad.
Para ello han recopilado una serie de imágenes que servirán como datos de entrada.

La empresa os pide:

* Realizar un análisis exploratorio de los datos detallando aquellos aspectos más relevantes que hayáis encontrado.
* Construir dos modelos: uno de regresión que prediga la edad de la persona de la foto (Podríamos cambiarlo en función a como se desarrolle el trabajo) y otro de clasificación que prediga si es hombre o mujer en la foto
* Desarrollar un cuadro de mando con Dash que resuma los aspectos más relevantes que hayáis extraido en el análisis exploratorio, muestre el proceso de modelado de los datos y permita evaluar fotos nuevas.

¿Qué tareas de mantenimiento o evolución de la aplicación son necesarias para que se mejore la aplicación realizada?

## Información de los datos

Los datos son fotos por lo tanto no hay variables explícitamente. Tenéis que saber que si una foto se llama de la siguiente forma __34_0_0_20170117120149082.jpg.chip.jpg__
significa que la edad del individuo es 34 y que su género es hombre. Es decir, los nombres de las fotos siguen el siguiente esquema __edad_género_raza_datosirrelevantes.jpg.chip.jpg__.

Siendo el género 0 para hombre y 1 para mujer.