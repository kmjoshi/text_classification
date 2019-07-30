import gradio


def predict(s):
    # s: single str object
    return {'good': 1}


io = gradio.Interface(inputs="textbox", outputs="label",
                      model_type="pyfunc", model=predict)
io.launch(inbrowser=True, share=False)
