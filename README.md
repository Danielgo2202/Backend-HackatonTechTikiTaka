# Backend-HackatonTechTikiTaka
# SignalCard
### Real-time Competitor Intelligence for Sales Calls

> **GTM Hackathon LatAm — Bogotá, Mayo 9–10, 2025 · 48 horas**
> 4 personas · 3 Ingenieros de Sistemas + 1 Ciencia de Datos · 25 puntos posibles · 12 jueces

---

## Índice

1. [Visión del producto](#1-visión-del-producto)
2. [Diferenciador competitivo](#2-diferenciador-competitivo)
3. [Cómo se usa](#3-cómo-se-usa)
4. [Stack tecnológico](#4-stack-tecnológico)
5. [Arquitectura del sistema](#5-arquitectura-del-sistema)
6. [Estructura del repositorio](#6-estructura-del-repositorio)
7. [Estructura de datos](#7-estructura-de-datos)
8. [División de trabajo — 48 horas](#8-división-de-trabajo--48-horas)
9. [Filosofía del MVP](#9-filosofía-del-mvp)

---

## 1. Visión del producto

### ¿Qué es SignalCard?

SignalCard es un copiloto de inteligencia competitiva en tiempo real para equipos de ventas. Escucha conversaciones activas y, en el momento exacto en que un cliente menciona a un competidor, despliega automáticamente una **battlecard contextual** con todo lo que el vendedor necesita para responder con confianza — sin pausas, sin búsquedas, sin improvisación.

> 🎯 **La pregunta central del hackathon**
> *"¿Lo usaría un equipo de ventas, marketing o RevOps el lunes?"*
> Si no — no cuenta.

### Problema que resuelve

Durante llamadas de ventas, los vendedores enfrentan estos problemas de forma recurrente:

- Olvidan información clave sobre competidores en el momento de mayor presión
- Responden lentamente o improvisan frente a objeciones competitivas
- Pierden tiempo buscando en CRM, Notion, Google o docs internos
- La información existe — pero está fragmentada y nunca disponible en tiempo real

El resultado es predecible: menor tasa de cierre, respuestas débiles, pérdida de confianza y deals perdidos.

SignalCard resuelve cuatro problemas simultáneamente:
- Fragmentación de contexto
- Sobrecarga cognitiva durante la llamada
- Pérdida de información en conversaciones críticas
- Falta de sales enablement en tiempo real

### La visión del producto

> *"Una capa de inteligencia contextual en tiempo real para conversaciones de ventas."*
> No es un chatbot. No es un dashboard.
> Es el contexto correcto, en el momento exacto.

---

## 2. Diferenciador competitivo

### Por qué SignalCard gana

La diferencia central frente a todas las herramientas existentes es una sola palabra: **timing**.

| Herramienta | Cuándo actúa | Impacto |
|---|---|---|
| Spiky.ai | Después de que termina la llamada | El daño ya está hecho |
| Gong | Post-llamada, análisis y coaching | No ayuda en el momento |
| **SignalCard** | **Mientras el cliente todavía está en línea** | **El vendedor responde en el acto** |

El intervalo entre detección y respuesta es de **menos de 10 segundos**. Eso no existe en ningún producto actual de LatAm.

### Criterios del hackathon — Proyección de puntaje

| Criterio | Qué evalúan | Cómo lo cubre SignalCard | Score esperado |
|---|---|---|---|
| Impact | Mueve revenue directamente | Mejora win rate en llamadas activas | 5 / 5 |
| Execution | Corre en vivo ante jueces | Demo con audio real o simulado, latencia < 10s | 5 / 5 |
| Creativity | No es Clay ni Apollo | Nada en LatAm hace esto en tiempo real | 4–5 / 5 |
| Automation | Reutilizable para cualquier empresa | Battlecards configurables por cliente | 4–5 / 5 |
| Presentation | Demo visual e inmediata | Momento WOW: card aparece al decir el competidor | 5 / 5 |

---

## 3. Cómo se usa

### Flujo de uso real

El vendedor no cambia su flujo de trabajo. SignalCard se integra como una pestaña paralela:

1. Abre Zoom o Google Meet y comienza la llamada normalmente
2. Abre SignalCard en una pestaña del navegador al lado
3. Comparte el audio de la reunión con la aplicación (`getDisplayMedia`)
4. La app transcribe la conversación en vivo en el panel central
5. En el momento en que el cliente menciona un competidor, la battlecard aparece automáticamente en el sidebar derecho
6. El vendedor lee el diferenciador, la sugerencia de respuesta y el contexto del cliente — y responde con confianza

> 💡 **Momento WOW para el demo**
> Cliente dice: *"También estamos evaluando HubSpot."*
> → 1 segundo →
> Battlecard de HubSpot aparece animada en pantalla
> → El vendedor responde: *"¿Qué limitaciones han encontrado con HubSpot hasta ahora?"*
> → El juez entiende el valor de inmediato.

### Layout de la interfaz

La interfaz es minimalista y moderna, inspirada en **Cursor, Linear, Perplexity y Arc Browser**.

| Zona | Contenido |
|---|---|
| Centro | Transcript en vivo de la conversación, actualizado en tiempo real |
| Sidebar derecho | Battlecards dinámicas: aparecen con animación suave, se apilan verticalmente, permanecen visibles |
| Cada battlecard | Diferenciador clave · Debilidades del competidor · Manejo de objeción · Pregunta sugerida |

---

## 4. Stack tecnológico

### Herramientas y decisiones

| Capa | Herramienta | Rol específico | Alternativa / Fallback |
|---|---|---|---|
| Captura de audio | MediaRecorder API (browser) | Captura audio de Zoom/Meet vía `getDisplayMedia` | Archivo `.wav` pregrabado para demo |
| Transporte de audio | WebSockets (nativo FastAPI) | Stream binario browser → backend en chunks de 1s | — |
| STT (Transcripción) | Deepgram Streaming API | Transcripción en tiempo real con baja latencia | Whisper API de OpenAI |
| Orquestación IA | LangChain (Python) | Cadena RetrievalQA: texto → Chroma → LLM | — |
| Vector Store | ChromaDB | Almacena battlecards como embeddings, búsqueda semántica | JSON local para MVP mínimo |
| LLM | OpenAI GPT-4o mini | Clasificación de intent + formateo de battlecard | GPT-4o si se requiere más contexto |
| Backend / API | FastAPI (Python) | WebSocket server + endpoints REST | — |
| Base de datos CRM | Supabase (PostgreSQL) | Contexto del cliente: nombre, industria, deal size | — |
| Frontend | Next.js + React + Tailwind | SPA moderna, manejo de estado, UI en tiempo real | Lovable para prototipo rápido |
| Animaciones | Framer Motion | Slide-in y fade-in de battlecards | — |
| Deploy frontend | Vercel | Deploy automático desde GitHub | — |
| Deploy backend | **Render (free tier)** | Deploy desde GitHub, sin tarjeta de crédito | Railway |

### Qué se eliminó del stack original y por qué

| Herramienta | Decisión | Razón |
|---|---|---|
| n8n / Make | ❌ Eliminado | Latencia extra, capa de debug innecesaria, no aporta al demo |
| Zoom API oficial | ❌ Eliminado para demo | WiFi del hackathon no es confiable, riesgo de fallo alto |
| SSE | → Reemplazado por WebSockets | La comunicación es bidireccional: el browser también envía audio |
| GPT-4o full | → GPT-4o mini | 2–3x menor latencia, suficiente para clasificar y formatear |
| Llamadas directas a OpenAI | → LangChain | Pipeline orquestado, más fácil de debuggear |
| Supabase para battlecards | → ChromaDB | Búsqueda semántica > string matching para detección de competidores |

---

## 5. Arquitectura del sistema

### Principio rector

> Simple · Realtime · Estable · Demoable.
> Una app frontend. Una app backend. Sin microservicios, sin Kafka, sin Redis, sin Kubernetes.

### Diagrama de arquitectura

```
[ Browser — Vendedor ]
    MediaRecorder → audio/webm chunks (1 segundo cada uno)
    WebSocket ↕ (conexión persistente y bidireccional)

[ FastAPI Backend — Render ]
    ├── Deepgram Streaming API  →  transcript text
    ├── LangChain RetrievalQA Chain
    │       ├── ChromaDB         →  battlecard por similarity search
    │       └── Supabase         →  contexto CRM del cliente activo
    └── GPT-4o mini              →  battlecard formateada y lista

    WebSocket → { "type": "battlecard", "competitor": "HubSpot", "data": { ... } }

[ React Frontend — Vercel ]
    socket.onmessage → addCard(data)
    Framer Motion    → slide-in animado en sidebar derecho
```

### Pipeline — Paso a paso

| # | Etapa | Detalle técnico |
|---|---|---|
| 01 | Captura de audio | El browser usa `navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })`. `MediaRecorder` crea chunks binarios cada 1 segundo en formato `audio/webm`. |
| 02 | Stream al backend | El frontend abre `const socket = new WebSocket("wss://backend/ws")`. Cada chunk se envía como binario: `socket.send(audioBlob)`. La conexión permanece abierta durante toda la llamada. |
| 03 | Transcripción | FastAPI recibe los bytes y los reenvía a Deepgram Streaming API. Deepgram devuelve transcript parcial: `{ "transcript": "We are also evaluating HubSpot" }`. Latencia < 1 segundo. |
| 04 | Detección de intent | LangChain procesa el texto. Primero intenta match por string simple (lista de competidores conocidos). Si no hay match exacto, usa embedding + ChromaDB para encontrar el competidor semánticamente más cercano. |
| 05 | Retrieval de battlecard | Chroma ejecuta similarity search con el texto del transcript. Retorna el documento más relevante (battlecard del competidor detectado) con embeddings pre-cargados al iniciar el servidor. |
| 06 | Contexto del cliente | Supabase retorna el contexto CRM del cliente activo: nombre, industria, tamaño del deal, pain points. Se une al resultado de Chroma antes de enviarlo al frontend. |
| 07 | GPT-4o mini | LangChain invoca GPT-4o mini para formatear la respuesta final: diferenciador principal, sugerencia de respuesta lista para decir, y pregunta recomendada. Latencia < 2 segundos. |
| 08 | Envío al frontend | FastAPI envía JSON event via WebSocket. El frontend escucha `socket.onmessage` y actualiza el estado React con `addCard(data)`. |
| 09 | Renderizado | Framer Motion anima la aparición de la card con slide-in desde la derecha + fade-in. Las cards se apilan en el sidebar y permanecen visibles hasta el fin de la llamada. |

### Decisiones técnicas justificadas

| Decisión | Justificación |
|---|---|
| WebSockets sobre SSE | La comunicación es bidireccional: el browser envía audio binario y el backend devuelve eventos JSON. SSE es unidireccional y no puede recibir el stream de audio. |
| ChromaDB sobre Supabase para battlecards | Chroma permite búsqueda semántica. Si el cliente dice "usamos el CRM de Salesforce" sin mencionar el nombre exacto, igual detecta la match. String matching fallaría. |
| LangChain como orquestador | Abstrae la cadena STT → clasificación → retrieval → LLM en un solo pipeline mantenible. Más fácil de debuggear que llamadas directas encadenadas a mano. |
| GPT-4o mini sobre full GPT-4o | Latencia 2–3x menor. Para clasificar texto y formatear una respuesta breve, el modelo mini es más que suficiente. Reduce costos en demo. |
| Audio simulado para demo | El WiFi de un hackathon no es confiable. Un `.wav` pregrabado garantiza que el pipeline completo corra sin depender de Zoom. Para el juez el efecto visual es idéntico. |
| Deepgram con fallback a Whisper | Deepgram tiene streaming real-time nativo. Si los créditos fallan, Whisper corre localmente o vía API con latencia aceptable. |
| Chroma inicializado al arrancar | Render free tier permite disco efímero. Los embeddings se cargan una vez en startup. Ningún request paga el costo de inicialización. |

---

## 6. Estructura del repositorio

```
signalcard/
├── backend/
│   ├── main.py                  # FastAPI app + WebSocket endpoint principal
│   ├── transcription.py         # Deepgram streaming client
│   ├── chain.py                 # LangChain RetrievalQA chain
│   ├── chroma_init.py           # Carga embeddings de battlecards al startup
│   ├── supabase_client.py       # Consultas al contexto CRM del cliente
│   ├── battlecards/             # Datos de battlecards (JSON fuente)
│   │   ├── hubspot.json
│   │   ├── salesforce.json
│   │   ├── gong.json
│   │   └── apollo.json
│   ├── chroma_db/               # Vector store persistido (generado en startup)
│   ├── requirements.txt
│   └── render.yaml              # Configuración de deploy en Render
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Layout principal (transcript + sidebar)
│   │   └── components/
│   │       ├── AudioCapture.tsx      # MediaRecorder + WebSocket client
│   │       ├── Transcript.tsx        # Panel central con texto en vivo
│   │       └── BattlecardSidebar.tsx # Stack de cards con Framer Motion
│   ├── package.json
│   └── vercel.json
│
└── README.md
```

### Responsabilidades clave por archivo

| Archivo | Qué hace |
|---|---|
| `main.py` | Recibe conexión WebSocket, pasa audio a `transcription.py`, llama a `chain.py`, devuelve evento al frontend |
| `transcription.py` | Abre stream con Deepgram, recibe chunks de audio y retorna texto parcial |
| `chain.py` | Construye el `RetrievalQA` chain de LangChain: Chroma retriever + GPT-4o mini |
| `chroma_init.py` | Lee los JSON de battlecards, genera embeddings y los carga en ChromaDB al inicio del servidor |
| `supabase_client.py` | Consulta la tabla `clients` en Supabase para obtener el contexto del cliente activo |
| `AudioCapture.tsx` | Pide permiso de pantalla, crea MediaRecorder, abre WebSocket y envía chunks cada segundo |
| `Transcript.tsx` | Recibe texto del estado React y lo muestra en tiempo real en el panel central |
| `BattlecardSidebar.tsx` | Escucha eventos `battlecard` del WebSocket, renderiza cards con Framer Motion |

---

## 7. Estructura de datos

### Formato de una battlecard (JSON fuente)

Cada battlecard es un JSON que se convierte en embedding y se almacena en ChromaDB antes de que el servidor reciba su primera request.

```json
{
  "competitor": "HubSpot",
  "strengths": [
    "Mejor personalización de workflows",
    "Menor costo para enterprise",
    "API más flexible y extensible"
  ],
  "weaknesses": [
    "Escalabilidad limitada a partir de 500 usuarios",
    "Add-ons costosos para funciones que deberían ser básicas"
  ],
  "suggested_response": "Muchos equipos migran desde HubSpot cuando los workflows se vuelven más complejos. ¿Han tenido ese problema?",
  "recommended_question": "¿Qué limitaciones han encontrado con HubSpot hasta ahora?",
  "key_differentiator": "Nuestro motor de automatización no cobra por acción — el de HubSpot sí."
}
```

### Evento WebSocket backend → frontend

```json
{
  "type": "battlecard",
  "competitor": "HubSpot",
  "confidence": 0.96,
  "data": {
    "key_differentiator": "Nuestro motor de automatización no cobra por acción.",
    "suggested_response": "Muchos equipos migran desde HubSpot cuando los workflows se vuelven más complejos.",
    "recommended_question": "¿Qué limitaciones han encontrado con HubSpot hasta ahora?",
    "weaknesses": [
      "Escalabilidad limitada",
      "Add-ons costosos"
    ]
  },
  "client_context": {
    "name": "Empresa ABC",
    "industry": "SaaS B2B",
    "deal_size": "$45,000"
  }
}
```

### Schema de Supabase

**Tabla `clients`**

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | uuid | PK |
| `name` | text | Nombre de la empresa cliente |
| `industry` | text | Industria (SaaS B2B, Fintech, etc.) |
| `deal_size` | text | Tamaño estimado del deal |
| `pain_points` | text[] | Array de pain points conocidos |
| `active` | boolean | Si es el cliente de la sesión actual |

### Competidores pre-cargados en Chroma (para demo)

| Competidor | Variaciones que detecta |
|---|---|
| HubSpot | "HubSpot", "Hub Spot", "el CRM de HubSpot" |
| Salesforce | "Salesforce", "Sales Force", "SFDC", "el CRM de Salesforce" |
| Gong | "Gong", "Gong.io", "la plataforma de Gong" |
| Apollo | "Apollo", "Apollo.io" |

---

## 8. División de trabajo — 48 horas

### Roles del equipo

| Persona | Rol | Horas 1–16 | Horas 17–48 |
|---|---|---|---|
| Ing 1 | Backend Lead | FastAPI WebSocket server + Deepgram client | Integración Chroma + optimización de latencia |
| Ing 2 | LangChain + IA | LangChain RetrievalQA chain + GPT-4o mini + Chroma init | Tuning del prompt + manejo de edge cases |
| Ing 3 | Frontend Lead | Next.js app + AudioCapture + WebSocket client | Sidebar animado + Framer Motion + polish de UI |
| Data Science | Datos + Embeddings | Crear y poblar battlecards JSON + generar embeddings | Supabase CRM context + validación end-to-end del pipeline |

### Timeline de 48 horas

| Bloque | Objetivos clave |
|---|---|
| H 1–4 | Repo setup · Render deploy vacío · Vercel deploy vacío · Supabase schema · Chroma init script corriendo |
| H 5–10 | WebSocket abierto browser ↔ backend · MediaRecorder enviando audio · Deepgram retornando transcript |
| H 11–16 | LangChain chain completa · Chroma con 4 battlecards cargadas · Detección funcionando end-to-end |
| H 17–24 | Frontend con transcript en vivo · Primera battlecard apareciendo en sidebar · Animación básica |
| H 25–36 | Polish de UI · Contexto CRM de Supabase integrado · Manejo de errores · Fallbacks |
| H 37–44 | Rehearsal del demo con audio `.wav` pregrabado · Ajuste de timing · Validación en Render |
| H 45–48 | Presentación · Buffer para imprevistos · Slides de apoyo si se necesitan |

### Dependencias críticas entre personas

```
Data Science  →  chroma_init.py necesita battlecards.json listos antes de H 11
Ing 2         →  chain.py depende de que Ing 1 tenga el WebSocket recibiendo texto (H 10)
Ing 3         →  BattlecardSidebar.tsx necesita el formato del evento WebSocket definido (H 8)
```

> **Acción antes de H 4:** definir y versionar el schema del evento WebSocket (`battlecard` event) para que Ing 2, Ing 3 y Data Science trabajen en paralelo sin bloquearse.

---

## 9. Filosofía del MVP

### Qué incluye y qué no incluye

**✅ SignalCard SÍ hace**
- Detecta eventos específicos: menciones de competidores durante llamadas en vivo
- Muestra contexto útil y pre-cargado al instante (< 10 segundos)
- Funciona con audio real capturado del browser o con un `.wav` pregrabado para demo
- Es reutilizable para cualquier empresa con sus propias battlecards
- Es rápido, preciso y estable — diseñado para no fallar en demo

**❌ SignalCard NO hace**
- Buscar información libremente en internet durante la llamada
- Generar respuestas largas o documentos
- Analizar llamadas post-facto (eso ya lo hace Gong/Spiky)
- Integrarse con Zoom vía API oficial (fuera del scope del hackathon)
- Resolver todos los problemas de sales enablement
apareció exactamente cuando el cliente mencionó HubSpot"*.
> Eso es lo que importa.

---

## Resumen ejecutivo en una línea

**SignalCard** escucha llamadas de ventas en tiempo real, detecta menciones de competidores con LangChain + ChromaDB, y muestra battlecards contextuales en menos de 10 segundos — mientras el cliente todavía está en línea.

---

*SignalCard · GTM Hackathon LatAm · Bogotá, Mayo 2025*
*El contexto correcto. En el momento exacto.*
### Reglas de arquitectura del MVP

- **No hacer:** microservicios, Kafka, Redis, queues, Kubernetes
- **Hacer:** una app backend, una app frontend, y listo
- **Prioridad:** que el demo no falle > que el código sea perfecto
- **Si algo falla en vivo:** el fallback es el `.wav` pregrabado — el pipeline completo igual corre

> ⚡ **Regla de oro**
> Los jueces nunca dirán *"wow, qué arquitectura"*.
> Dirán *"wow, eso 
