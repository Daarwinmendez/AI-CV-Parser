Motor automatizado de extracción de perfiles profesionales, diseñado para convertir hojas de vida (currículums) no estructuradas en objetos de datos tipados y listos para producción.

## 📌 Caso de Negocio
En el entorno empresarial y de reclutamiento, gran parte de la información crítica se encuentra en formatos no estructurados (documentos PDF, imágenes, texto plano). Este sistema implementa una arquitectura moderna para la normalización de dichos datos, reduciendo el esfuerzo operativo manual y eliminando la variabilidad humana en la fase de ingesta.

El sistema procesa hojas de vida y extrae entidades clave bajo un esquema estricto de validación tipográfica.

##  Arquitectura del Pipeline
El flujo de ejecución de extracción estructurada consta de tres etapas deterministas:

1. **Preprocesamiento:** Ingesta del documento PDF y limpieza tipográfica mediante expresiones regulares (RegEx).
2. **Inferencia Determinista:** El texto se procesa utilizando un motor de razonamiento (`Meta-Llama-3-8B-Instruct`) configurado sin creatividad (`temperature=0.0`) para anular alucinaciones y extraer entidades fidedignas.
3. **Validación Estricta:** La salida se somete a tipado estricto con Pydantic, garantizando que cada valor extraído (habilidades, experiencias, proyectos destacados) cumpla con el esquema exacto de datos antes de ser procesado o almacenado en bases relacionales o vectoriales.

## Requisitos e Instalación

1. Clona el repositorio:
```bash
   git clone [https://github.com/Daarwinmendez/AI-CV-Parser.git](https://github.com/Daarwinmendez/AI-CV-Parser.git)
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura tus credenciales creando un archivo `.env` en la raíz del proyecto con tu API Token de Hugging Face:
```env
HUGGINGFACEHUB_API_TOKEN=tu_token_aqui
```


## 🚀 Uso y Arquitectura

El pipeline se ejecuta dentro del notebook principal ubicado en `notebooks/CV_Parser.ipynb`. En él se carga el modelo de lenguaje, se inicializan los esquemas de Pydantic, se procesa el documento PDF mediante funciones optimizadas de PyMuPDF, y se valida la salida cruda a formato JSON estructurado. Desarrollado como solución automatizada de inteligencia artificial.
