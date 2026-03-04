from langchain_community.document_loaders import PyMuPDFLoader


loader = PyMuPDFLoader("./pdf-file.pdf")


docs = loader.load()

print(docs) # 업로드한 파일이 총 3개의 페이지가 있기 때문에 총 3개의 Document 객체가 생성됨

print(len(docs))

print(docs[0].metadata)