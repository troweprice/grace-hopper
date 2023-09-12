from contextlib import asynccontextmanager
import structlog as logging
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel

from src.config import get_config_for_model_download
from src.dependencies import get_document_sourcer, get_prompt_builder, get_retriever_on_app_start
from dotenv import load_dotenv

from src.prompt.prompt_builder import PromptBuilder

logger = logging.get_logger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and vector db on startup and close on shutdown."""
    # download the model specified in the configurations and save it to the app state
    model_config = get_config_for_model_download()
    retriever = get_retriever_on_app_start()
    app.state.MODEL = retriever.get_model(model_config)

    # create a vector db with documents and save it to the app state
    document_sourcer = get_document_sourcer()
    app.state.DB = document_sourcer.get_preloaded_vector_database()
    yield


# create a FastAPI instance to start up the app
app = FastAPI(lifespan=lifespan)


class UserQuery(BaseModel):
    query: str


# create a POST endpoint for model inference
@app.post("/api/v1/inference")
def inference(user_query: UserQuery, prompt_builder: PromptBuilder = Depends(get_prompt_builder)
              ):
    """Generate response for user query."""
    #  retrieve the model from app state
    model = app.state.MODEL

    # if a request is received before the model loading is complete
    if model is None:
        raise HTTPException(status_code=status.HTTP_425_TOO_EARLY, detail="Model not loaded")

    # if a request is received without a query
    query = user_query.query
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty")

    try:
        #  get a prompt for the model using the user query and document database
        chroma_client = app.state.DB
        engineered_prompt = prompt_builder.get_complete_prompt(query, chroma_client)

        #  send the prompt to the model
        config = get_config_for_model_download()
        responses = model(
            engineered_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )["choices"]

        # format the model response
        concatenated_response = "".join([response["text"] for response in responses])
        logger.info(f"Generated response: {concatenated_response}")

        # return the response to the user
        return concatenated_response.strip()

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response") from e
