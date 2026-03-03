from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # sweet spot for this doc
    chunk_overlap=120,     # helps preserve context
    separators=[
        "\n\n",            # section breaks
        "\n",              # line breaks
        ".",               # sentences
        " ",               # fallback
    ]
)