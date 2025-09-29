from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, StringField
from wtforms.validators import DataRequired, Length


class OriginalTextForm(FlaskForm):
	original_text = TextAreaField('Original Text', validators=[Length(min=20, max=10000)], render_kw={'placeholder': 'Enter text here..'})
	generate = SubmitField('Generate Text')
	predict = SubmitField('Predict')
	url = StringField('Article URL', render_kw={'placeholder': 'Paste article URL (optional)'})
	fetch_url = SubmitField('Fetch from URL')
