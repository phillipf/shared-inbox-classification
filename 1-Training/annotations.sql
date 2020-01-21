SELECT
text.id 
,text.text
,label_text.text as label
FROM public.api_document text
LEFT JOIN public.api_documentannotation label_id on text.id = label_id.document_id
LEFT JOIN public.api_label label_text on label_id.label_id = label_text.id
