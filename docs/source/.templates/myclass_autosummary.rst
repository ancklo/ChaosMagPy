{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}

   .. rubric:: Methods

   .. autosummary::
      ~{{ name }}.__call__
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   .. automethod:: __call__

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   {% else %}
   .. automethod:: __init__

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
