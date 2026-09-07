import requests

class GENE_LOOKUP():
	server="https://rest.ensembl.org"

	def name(id):
		ext=f"/lookup/id/{id}?expand=0"

		r=requests.get(GENE_LOOKUP.server + ext, headers={"Content-Type": "application/json"})

		if not r.ok:
			return None

		return r.json().get("display_name", None)

	def id(name):
		endpoint=f"/lookup/symbol/homo_sapiens/{name}?expand=1"

		r = requests.get(GENE_LOOKUP.server + endpoint, headers={"Content-Type": "application/json"})

		if not r.ok:
			return None

		return r.json().get('id')