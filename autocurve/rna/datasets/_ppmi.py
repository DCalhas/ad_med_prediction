import numpy as np

import pandas as pd

import os

from autocurve.rna.datasets import _Dataset, _GetTransitions, preprocess_step_curve

__all__=["PPMI"]

def get_ppmi(dataset_dir="/shared/home/david.calhas/ppmi", experiment="IR3"):
	assert experiment in ["IR2", "IR3"]
	
	rna_files=sorted([f for f in os.listdir(dataset_dir+"/rna/quant") if "genes" in f and experiment in f])
	
	df=pd.read_csv(dataset_dir+"/rna/quant/"+rna_files[0], sep="\t")
	gene_ind_matrix=np.empty(((4756, 58294) if experiment=="IR3" else (4755,34569)), dtype=np.float32)

	genes=df["Name"].to_numpy()
	individuals=[]

	for ind in range(len(rna_files)):
		df=pd.read_csv(dataset_dir+"/rna/quant/"+rna_files[ind], sep="\t")
		individuals.append(rna_files[ind].split(".")[1])

		gene_ind_matrix[ind, ]=df["TPM"]
		if(ind%10==0):
			print("I: "+str(ind+1)+" out of "+str(len(rna_files)), end="\r")

	print("I: Successfully loaded gene by individuals matrix.")

	tpm=df["TPM"]

	genes_ids=np.array([g.split(".")[0] for g in genes])

	PAT_INFO=pd.read_csv(dataset_dir+"/clinical/"+"Participant_Status_08Oct2025.csv", )
	PD_PATIENTS=PAT_INFO[PAT_INFO["COHORT"]==1]["PATNO"].to_numpy()
	HC_PATIENTS=PAT_INFO[PAT_INFO["COHORT"]==2]["PATNO"].to_numpy()
	individuals=np.array(individuals,)
	
	hc_rows=gene_ind_matrix[np.isin(individuals, HC_PATIENTS),]
	hc_target=np.zeros(hc_rows.shape[0])

	pd_rows=gene_ind_matrix[np.isin(individuals, PD_PATIENTS),]
	pd_target=1*np.ones(pd_rows.shape[0])

	individuals=np.concatenate((individuals[np.isin(individuals, HC_PATIENTS)], individuals[np.isin(individuals, PD_PATIENTS)]))

	hc_pd_rows=np.concatenate((hc_rows,pd_rows), axis=0)
	hc_pd_target=np.expand_dims(np.concatenate((hc_target,pd_target), axis=0), axis=1)

	genes_names=np.load(dataset_dir+"/rna/genes_names.npy", ).astype(genes_ids.dtype)
	
	return hc_pd_rows, genes_ids, genes_names, individuals.astype("U25"), hc_pd_target[:,0]

class Curve(np.ndarray):
	"""
	Class for curve representation
	a curve has a t for timesteps and then it can have the measured values for p1, p1p, p2, p3 and hoehn&yahr score
	"""

	types=["p1", "p1p", "p2", "p3", "hy"]

	def __new__(cls, patno, t, birth, gender, p1=None, p1p=None, p2=None, p3=None, hy=None, state_dbs=None, after_dbs=None, ttol=4, **kwargs):

		exists=[p1 is not None and len(p1)>=ttol, p1p is not None and len(p1p)>=ttol, p2 is not None and len(p2)>=ttol, p3 is not None and len(p3)>=ttol, hy is not None and len(hy)>=ttol]
		
		if(np.all(np.logical_not(exists))): return None
		
		#p1 = p1.astype(np.float32) if exists[0] else -1*np.ones((t.shape[0]))
		#p1p = p1p.astype(np.float32) if exists[1] else -1*np.ones((t.shape[0]))
		#p2 = p2.astype(np.float32) if exists[2] else -1*np.ones((t.shape[0]))
		p3 = p3.astype(np.float32) if exists[3] else -1*np.ones((t.shape[0]))
		hy = hy.astype(np.float32) if exists[4] else -1*np.ones((t.shape[0]))

		birth=birth.to_numpy()
		if(birth.shape and birth.shape[0]>0):
			birth=birth[0]
		else:
			birth=t.to_numpy()#zero imputation

		if(gender.shape and gender.shape[0]>0):
			gender=gender.to_numpy()[0].astype(np.int32)
		else:
			gender=-1

		#_t=np.cumsum(np.concatenate((np.zeros((1,)), np.diff(t).astype(np.timedelta64(1, 'W')).astype(np.float32))))

		#obj=np.asarray(np.vstack([_t, visit_age_weeks, p1, p1p, p2, p3, hy])).view(cls)
		obj=np.asarray(np.vstack([t, p3, hy])).view(cls)

		hy=np.nan_to_num(hy)
		hy[np.where(hy==101)]=0

		#preprocess curves
		#_, visit_age_weeks=preprocess_step_curve(hy, visit_age_weeks, values=State.values)
		hy, t, keep = preprocess_step_curve(
			hy, t, values=State.values, return_indices=True, method="cummax",
		)

		obj.patno=patno
		obj.date=t.to_numpy().astype("datetime64[D]")
		obj.t=t
		obj.state_dbs=state_dbs # state \in \{ 0,1,2,3 \} 0 - off med off stim; 1 - on med off stim; 2 - off med on stim; 3 - on med on stim
		obj.after_dbs=after_dbs
		obj.gender=gender
		#obj.p1=p1
		#obj.p1p=p1p
		#obj.p2=p2
		obj.p3=p3
		obj.progression=hy
		types=dict(zip(Curve.types, exists))
		obj.types=types
		
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.patno = getattr(obj, "patno", None)
		self.t = getattr(obj, "t", None)
		self.date=getattr(obj, "date", None)
		self.state_dbs=getattr(obj, "state_dbs", None)
		self.after_dbs=getattr(obj, "after_dbs", None)
		self.gender=getattr(obj, "gender", None)
		#self.p1 = getattr(obj, "p1", None)
		#self.p1p = getattr(obj, "p1p", None)
		#self.p2 = getattr(obj, "p2", None)
		self.p3 = getattr(obj, "p3", None)
		self.progression = getattr(obj, "progression", None)
		self.types = getattr(obj, "types", None)

	def getxy(self, curve_type="p1"):
		
		assert self.types[curve_type]

		return self.t, getattr(self, curve_type)

visits_names=["BL", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25"]
visits_month=dict(zip(visits_names, [0,3,6,9,12,18,24,30,36,42,48,54,60,72,84,96,108,120,132,144,156,168,180,192,204,216,]))

remote_visits_names=["R01","R04","R06","R08","R10","R12","R13","R14","R15","R16","R17","R18","R19","R20","R21","R22","R23","R24",]
visits_month.update(dict(zip(remote_visits_names,[6,18,30,42,54,66,78,90,102,114,126,138,150,162,174,186,198,210,])))

def _get_curves(dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/ppmi", state="ON"):
	
	INFO=pd.read_csv(dataset_dir+"/clinical/Participant_Status_08Oct2025.csv")
	DEMOG=pd.read_csv(dataset_dir+"/clinical/Demographics_21Jan2026.csv", sep=",")
	PD_PATNO=INFO[INFO["COHORT"]==1]["PATNO"]
	
	P1=pd.read_csv(dataset_dir+"/clinical/MDS-UPDRS_Part_I_11Feb2026.csv")
	P1Q=pd.read_csv(dataset_dir+"/clinical/MDS-UPDRS_Part_I_Patient_Questionnaire_11Feb2026.csv")
	P2=pd.read_csv(dataset_dir+"/clinical/MDS_UPDRS_Part_II__Patient_Questionnaire_11Feb2026.csv")
	P3=pd.read_csv(dataset_dir+"/clinical/MDS-UPDRS_Part_III_11Feb2026.csv")

	for col, df in [('NP1RTOT', P1), ('NP1PTOT', P1Q), ('NP2PTOT', P2), ('NP3TOT', P3), ('NHY', P3)]:
		df[col] = pd.to_numeric(df[col], errors='coerce')
	
	P1_sub = P1[['PATNO', 'EVENT_ID', 'NP1RTOT']]
	P1Q_sub = P1Q[['PATNO', 'EVENT_ID', 'NP1PTOT']]
	P2_sub = P2[['PATNO', 'EVENT_ID', 'NP2PTOT']]
	P3_sub = P3[['PATNO', 'EVENT_ID', 'NP3TOT', 'NHY']]
	
	for df in [P3]:
		df=df.copy()
		df=df[df["EVENT_ID"].isin(visits_month)]
		df["EVENT_ID"]=pd.to_numeric(df["EVENT_ID"].map(visits_month))
	
	procedures=pd.read_csv(dataset_dir+"/clinical/Procedure_for_PD_Log_26Jan2026.csv")
	procedures=procedures[procedures['PDSURGTP']==1]	
	procedures=procedures[procedures["EVENT_ID"].isin(visits_month)]
	procedures["EVENT_ID"]=pd.to_numeric(procedures["EVENT_ID"].map(visits_month))

	#get dbs data
	P3["NHY_CLEAN"] = P3["NHY"].where(P3["NHY"] != 101, np.nan)

	dbson=P3[P3["HRDBSON"].notna()].copy()
	dbsoff=P3[P3["HRDBSOFF"].notna()].copy()
	dbsonmed=dbson[dbson["PDSTATE"].isin(["ON", "OFF"])].copy()
	dbsoffmed=dbsoff[dbsoff["PDSTATE"].isin(["ON", "OFF"])].copy()
	dbsonmed["MED_STATE"]=dbsonmed["PDSTATE"]
	dbsoffmed["MED_STATE"]=dbsoffmed["PDSTATE"]
	dbsonmed["DBS_STATE"] = np.where(dbsonmed["HRDBSON"].notna(), "ON",np.where(dbsonmed["HRDBSOFF"].notna(), "OFF", np.nan))
	dbsoffmed["DBS_STATE"] = np.where(dbsoffmed["HRDBSON"].notna(), "ON",np.where(dbsoffmed["HRDBSOFF"].notna(), "OFF", np.nan))

	dbsonmed=dbsonmed[dbsonmed["DBS_STATE"]=="ON"]
	dbsoffmed=dbsoffmed[dbsoffmed["DBS_STATE"]=="OFF"]

	#print("ON MED, ON DBS", dbsonmed[dbsonmed["MED_STATE"]=="ON"]["PATNO"].unique().shape)
	#print("ON MED, OFF DBS", dbsoffmed[dbsoffmed["MED_STATE"]=="ON"].shape)
	#print("OFF MED, ON DBS", dbsonmed[dbsonmed["MED_STATE"]=="OFF"].shape)
	#print("OFF MED, OFF DBS", dbsoffmed[dbsoffmed["MED_STATE"]=="OFF"]["PATNO"].unique().shape)
	# medication state
	updrs3=P3[P3["PAG_NAME"]=="NUPDRS3"].copy()
	updrs3=updrs3[updrs3["PDSTATE"].isin(["ON", "OFF"])].copy()
	updrs3["MED_STATE"] = updrs3["PDSTATE"]

	updrs3["DBS_STATE"] = np.where(updrs3["HRDBSON"].notna(), "ON",np.where(updrs3["HRDBSOFF"].notna(), "OFF", np.nan))
	# clean HY: treat 101 as missing
	updrs3["NHY_CLEAN"] = updrs3["NHY"].where(updrs3["NHY"] != 101, np.nan)

	updrs3=updrs3[updrs3["MED_STATE"]==state].copy()

	updrs3=updrs3.sort_values(by=['PATNO', 'EVENT_ID'])

	updrs3=updrs3.dropna(subset=["NHY_CLEAN"])
	updrs3=updrs3.dropna(subset=["NP3TOT"])

	pat_curves={}
	
	for patno in PD_PATNO:
		IND_DEMOG=DEMOG[DEMOG['PATNO']==patno]
		birth=pd.to_datetime(IND_DEMOG["BIRTHDT"], format='%m/%Y', errors='coerce')
		
		gender=IND_DEMOG["SEX"]

		# After you've created result (patient-specific already)
		keys = ["EVENT_ID"]

		p3_patient = updrs3[updrs3["PATNO"] == patno].copy()
		p3_patient["EVENT_ID"]=pd.to_numeric(p3_patient["EVENT_ID"].map(visits_month))

		# Now these have the SAME length:
		part3 = p3_patient["NP3TOT"].to_numpy()
		state_dbs = p3_patient["DBS_STATE"].to_numpy()=="ON"

		if(~np.any(procedures["PATNO"]==patno)): 
			state_dbs=None 
			dbs_followup=None 
		else: 
			dbsdate=procedures[procedures["PATNO"]==patno]["EVENT_ID"].unique()[0]
			dbs_followup=(p3_patient["EVENT_ID"] >= dbsdate).to_numpy()

		patno=str(patno)

		#curve=Curve(patno, result_aligned["INFODT"], birth, gender, result_aligned["Part 1"].to_numpy(), result["Part 1P"].to_numpy(), result_aligned["Part 2"].to_numpy(), result_aligned["Part 3"].to_numpy(), result_aligned["H & Y Stage"].to_numpy(), state_dbs=state_dbs, after_dbs=dbs_followup, ttol=1)
		curve=Curve(patno, p3_patient["EVENT_ID"], birth, gender, None, None, None, p3_patient["NP3TOT"].to_numpy(), p3_patient["NHY_CLEAN"].to_numpy(), state_dbs=state_dbs, after_dbs=dbs_followup, ttol=1)
		if(curve is not None): pat_curves[patno]=curve
		print("I:",patno, "of", PD_PATNO.iloc[-1], end="\r")
	
	
	print("I: Successfully read progression curves.", end="\n")

	return pat_curves

class State():
	states=[0,1,2,3,4,5]
	values=[0,1,2,3,4,5]

	def __init__(self, x):
		assert x in State.states

		self.x=np.int32(x)

		if(self.x.shape): self.x=self.x[0]

	def __repr__(self,):

		return str(self.x)

class Transition(np.ndarray):

	def __new__(cls, patno : np.int32, duration : np.int32, gender : np.int32, orig : State, dest : State, time : np.int32):

		obj=np.asarray([duration, gender, orig.x, dest.x, time]).view(cls).astype(np.int32)

		obj.patno=patno
		obj.duration=duration
		obj.gender=gender
		obj.orig=orig.x
		obj.dest=dest.x
		obj.t=time
		obj.event=int(orig.x!=dest.x)
		
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.patno=getattr(obj, "patno", None)
		self.duration=getattr(obj, "duration", None)
		self.gender=getattr(obj, "gender", None)
		self.orig=getattr(obj, "orig", None)
		self.dest=getattr(obj, "dest", None)
		self.t=getattr(obj, "t", None)
		self.event=getattr(obj, "event", None)


ppmi_transitions=_GetTransitions(Transition, State, _get_curves)

class PPMI(_Dataset):

	get_dataset=get_ppmi
	curve=Curve
	state=State
	transition=Transition
	get_transitions=ppmi_transitions