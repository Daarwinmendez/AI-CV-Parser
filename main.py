# -*- coding: utf-8 -*-

import os
import re
import json
import torch
import fitz  # PyMuPDF
from typing import List, Optional
from google.colab import userdata

# LangChain & Transformers Imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

def preprocess_resume_text(text):
    """
    Toma un texto y lo limpia.
    Recibe:
      - text:str
    retorna:
      - text:str
    """

    text = text.replace("´", "")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_text_from_pdf(path):
    """
    Extrae texto de un PDF.
    Recibe:
      - path:str
    retorna:
      - text:str
    """

    try:
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return preprocess_resume_text(full_text)
    except Exception as e:
        return f"Error leyendo PDF: {e}"


# CONFIGURACIÓN DEL MODELO (OPTIMIZADO PARA BAJOS RECURSOS)

# Usamos Qwen 2.5 1.5B. Es MUY ligero y bien inteligente.
# Si tenemos un poco más de RAM, podemos probar "Qwen/Qwen2.5-3B-Instruct"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Obtener token
try:
    HF_TOKEN = userdata.get("HUGGINGFACEHUB_API_TOKEN")
except:
    HF_TOKEN = None # USAR OS GET ENV SI NO ES COLAB # Si no usamos secrets de colab

llm = HuggingFaceEndpoint(
    repo_id = model_id,
    huggingfacehub_api_token = HF_TOKEN,
    task = "text-generation",
    max_new_tokens = 2048, # Suficiente para un JSON de CV
    temperature = 0.0,
    repetition_penalty = 1.1
)

chat_model = ChatHuggingFace(llm=llm)

print("Modelo cargado exitosamente!")


class WorkExperience(BaseModel):
    position: str = Field(description="Cargo o puesto")
    company: str = Field(description="Nombre de la empresa")
    dates: Optional[str] = Field(description="Fechas trabajadas")

class Education(BaseModel):
    degree: str = Field(description="Título obtenido")
    institution: str = Field(description="Institución educativa")
    graduation_year: Optional[int] = Field(description="Año de graduación")


class FeaturedProjects(BaseModel):
   #Title y description siguen siendo obligatorios Si existe un proyecto.
    title: str = Field(description="El título del proyecto")
    description: str = Field(description="Una breve descripción")
    link: Optional[str] = Field(default=None, description="URL opcional")
    tech_stack: List[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    name: Optional[str] = Field(description="Nombre completo")
    email: Optional[str] = Field(default=None, description="Email")
    phone: Optional[str] = Field(default=None, description="Teléfono") # Default None
    skills: List[str] = Field(default_factory=list, description="Skills")
    work_experiences: List[WorkExperience] = Field(default_factory=list)
    education_history: List[Education] = Field(default_factory=list)
    featured_projects: List[FeaturedProjects] = Field(default_factory=list, description="Proyectos")

def preprocess_resume_text(text):
    text = text.replace("´", "")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return preprocess_resume_text(full_text)
    except Exception as e:
        return f"Error leyendo PDF: {e}"

#PROMPT Y CADENA

# El prompt está ajustado para modelos pequeños: Golden Rule: ser explícito y hacer tecnica del 1-shot

#PROMPT Y CADENA
resume_template = """<|im_start|>system
Eres un asistente de extracción de datos JSON.
Tu única tarea es convertir el CV en JSON siguiendo el esquema exacto proporcionado.
<|im_end|>
<|im_start|>user
Extrae la información del siguiente CV.

REGLAS DE VERDAD (ANTI-ALUCINACIONES):
1. No inventes títulos universitarios. Si no dice "Associate’s Degree" u otro título explícito, no lo pongas o pon el real.
2. Si falta un dato (como el teléfono o usuario de un link), tienes prohibido poner un nombre o link inventado, en su lugar si no está ahí, pon null.

REGLAS DE ESTRUCTURA (IMPORTANTE):
- Usa EXACTAMENTE las claves del ejemplo JSON de abajo (ej: usa "position", NO uses "title" o "role").
- graduation_year debe ser un número (2025) o null.

Ejemplo del Formato JSON Requerido:
{{
  "name": "Nombre Completo",
  "email": "correo@ejemplo.com",
  "phone": "123-456-7890",
  "skills": ["Python", "SQL"],
  "work_experiences": [
    {{
      "position": "Nombre del Cargo",
      "company": "Nombre Empresa",
      "dates": "Jan 2020 - Present"
    }}
  ],
  "education_history": [
    {{
      "degree": "Título Exacto",
      "institution": "Universidad",
      "graduation_year": 2025
    }}
  ],
  "featured_projects": [
    {{
      "title": "Título del Proyecto",
      "description": "Breve Descripción del Proyecto",
      "link": "URL del Proyecto",
      "tech_stack": ["Tech1", "Tech2"]
    }}
  ]
}}

--- TEXTO DEL CV ---
{resume_data}
--- FIN DEL CV ---
<|im_end|>
<|im_start|>assistant
"""

# Inicializamos el PromptTemplate explícitamente solo con la variable input
prompt = PromptTemplate(
    input_variables=["resume_data"],
    template=resume_template
)

def parse_resume(pdf_path: str):
    """
    Función para extraer CV de texto plano.
    """
    print(f"Leyendo PDF: {pdf_path}")
    resume_text = extract_text_from_pdf(pdf_path)

    if not resume_text or len(resume_text) < 50:
        print("El PDF parece estar vacío o no se pudo leer.")
        return None

    print("Procesando con LLM (Instrucciones estrictas)...")
    try:
        chain = prompt | chat_model
        response = chain.invoke({"resume_data": resume_text})
        content = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"Error en la inferencia del modelo: {e}")
        return None

    print("3. Extrayendo y validando JSON...")
    try:
        # Buscamos de forma no codiciosa el primer bloque delimitado por llaves { ... }
        json_match = re.search(r'\{.*\}', content, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)

            # Reemplazos formales para evitar que saltos de línea o comillas simples rompan el JSON
            json_str = json_str.replace("'", '"') # Cambiar comillas simples por dobles si las hay
            json_str = re.sub(r'[\n\r]+', ' ', json_str) # Aplanar saltos de línea
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str) # Eliminar comas sueltas (trailing commas)

            # Cargamos el diccionario limpio
            data_dict = json.loads(json_str)

            # Validación estricta con Pydantic
            profile = CandidateProfile(**data_dict)
            return profile
        else:
            print("ERROR: No se encontró un JSON válido en la respuesta.")
            print("Respuesta cruda:", content)
            return None

    except json.JSONDecodeError as e:
        print(f"ERROR JSON: El modelo generó un JSON malformado: {e}")
        print("Cadena problemática:", json_str if 'json_str' in locals() else content)
        return None
    except Exception as e:
        print(f"ERROR VALIDACIÓN: {e}")
        return None

extract_text_from_pdf("CV-Darwin Mendez - Jan-2026-EN.pdf")[280:590]

resultado=parse_resume(
    pdf_path = "CV-Darwin Mendez - Jan-2026-EN.pdf"
)

if resultado:
  print(resultado.model_dump_json(indent=2))