from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("./pdf-file.pdf")


docs = loader.load()

print(docs) # 업로드한 파일이 총 3개의 페이지가 있기 때문에 총 3개의 Document 객체가 생성됨

print(docs[0].page_content)