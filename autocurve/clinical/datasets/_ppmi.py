from functools import reduce

import numpy as np

import pandas as pd

import os

from autocurve.clinical.datasets import _Dataset, _GetTransitions, lnds_indices, preprocess_step_curve, nearest_interp_dates

from autocurve.rna.datasets._ppmi import get_ppmi

__all__=["PPMI"]

class State():
	states=[0,1,2,3,4,5]
	values=[0,1,2,3,4,5]

	def __init__(self, x):
		assert x in State.states

		self.x=np.int32(x)

		if(self.x.shape): self.x=self.x[0]

	def __repr__(self,):

		return str(self.x)

class Curve(np.ndarray):
	"""
	Class for curve representation
	a curve has a t for timesteps and then it can have the measured values for p1, p1p, p2, p3 and hoehn&yahr score
	"""

	types=['NP1COG', 'NP1HALL', 'NP1DPRS', 'NP1ANXS', 'NP1APAT', 'NP1DDS', 'NP1SLPN','NP1SLPD', 'NP1PAIN', 'NP1URIN', 'NP1CNST', 'NP1LTHD', 'NP1FATG', 'NP2SPCH','NP2SALV', 'NP2SWAL', 'NP2EAT', 'NP2DRES', 'NP2HYGN', 'NP2HWRT', 'NP2HOBB','NP2TURN', 'NP2TRMR', 'NP2RISE', 'NP2WALK', 'NP2FREZ', 'NP3SPCH', 'NP3FACXP','NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL', 'NP3FTAPR','NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL', 'NP3TTAPR','NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3RISNG', 'NP3GAIT', 'NP3FRZGT','NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', 'NP3PTRML', 'NP3KTRMR','NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL', 'NP3RTALJ', 'NP3RTCON', 'NP4WDYSK', 'NP4DYSKI', 'NP4OFF', 'NP4FLCTI', 'NP4FLCTX', 'NP4DYSTN']
	
	def __new__(cls, patno, t, birth, gender, updrs_features, updrs_values, hy=None, ttol=4, **kwargs):
		
		updrs_features=updrs_features
		updrs_values = updrs_values.astype(np.int32)
		hy = hy.astype(np.float32)

		birth=birth.to_numpy()
		if(birth.shape and birth.shape[0]>0):
			birth=birth[0]
		else:
			birth=t.to_numpy()

		visit_age_weeks=(t.to_numpy()-birth).astype(np.timedelta64(1, 'W')).astype(np.float32)

		if(gender.shape and gender.shape[0]>0):
			gender=gender.to_numpy()[0].astype(np.int32)
		else:
			gender=-1

		_t=np.cumsum(np.concatenate((np.zeros((1,)), np.diff(t).astype(np.timedelta64(1, 'W')).astype(np.float32))))

		#obj=np.asarray(np.vstack([_t, visit_age_weeks, p1, p1p, p2, p3, hy])).view(cls)
		obj=np.asarray(np.vstack([updrs_values])).view(cls)

		hy=np.nan_to_num(hy, nan=0)
		hy[np.where(hy==101)]=0

		#preprocess curves
		hy, _t, keep = preprocess_step_curve(
			hy, _t, values=State.values, return_indices=True, method="cummax",
		)
		visit_age_weeks = visit_age_weeks[keep]
		updrs_values = updrs_values[keep]
		dates = t.to_numpy().astype("datetime64[D]")[keep]

		obj.patno=patno
		obj.date=dates
		obj.t=_t
		obj.age_weeks=visit_age_weeks
		obj.gender=gender
		obj.cond_features=updrs_features	
		obj.cond_measures=updrs_values
		obj.progression=hy
		types=dict(zip(Curve.types, len(Curve.types)*[True]))
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
		self.age_weeks=getattr(obj, "age_weeks", None)
		self.gender=getattr(obj, "gender", None)
		#self.p1 = getattr(obj, "p1", None)
		#self.p1p = getattr(obj, "p1p", None)
		#self.p2 = getattr(obj, "p2", None)
		self.cond_measures = getattr(obj, "cond_measures", None)
		self.progression = getattr(obj, "progression", None)
		self.types = getattr(obj, "types", None)

	def getxy(self, curve_type="p1"):
		
		assert self.types[curve_type]

		return self.t, getattr(self, curve_type)


def _get_curves(dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/ppmi", state="ON"):
	
	INFO=pd.read_csv(dataset_dir+"/clinical/Participant_Status_08Oct2025.csv")
	DEMOG=pd.read_csv(dataset_dir+"/clinical/Demographics_21Jan2026.csv", sep=",")
	PD_PATNO=INFO[INFO["COHORT"]==1]["PATNO"]
	
	P1=pd.read_csv(dataset_dir+"/clinical/MDS-UPDRS_Part_I_11Feb2026.csv")
	P1P=pd.read_csv(dataset_dir+"/clinical/MDS-UPDRS_Part_I_Patient_Questionnaire_11Feb2026.csv")
	P2=pd.read_csv(dataset_dir+"/clinical/MDS_UPDRS_Part_II__Patient_Questionnaire_11Feb2026.csv")
	P3=pd.read_csv(dataset_dir+"/clinical/MDS-UPDRS_Part_III_11Feb2026.csv")
	P4=pd.read_csv(dataset_dir+"/clinical/MDS-UPDRS_Part_IV__Motor_Complications_11Feb2026.csv")


	#updrs1=P1[P1["PAG_NAME"]=="NUPDRS1"].copy()
	updrs1=P1.copy()
	updrs1=updrs1.sort_values(by=['INFODT', 'EVENT_ID'])
	updrs1=updrs1.drop(columns=updrs1.columns.difference([col for col in updrs1.columns if ("NP1" in col and not "NP1RTOT" in col) or col in ["PATNO", "INFODT", ]])).copy()
	p1features=[col for col in updrs1.columns if "NP1" in col and not "NP1RTOT" in col]

	#updrs1p=P1P[P1P["PAG_NAME"]=="NUPDRS1P2P"].copy()
	updrs1p=P1P.copy()
	updrs1p=updrs1p.sort_values(by=['INFODT', 'EVENT_ID'])
	updrs1p=updrs1p.drop(columns=updrs1p.columns.difference([col for col in updrs1p.columns if ("NP1" in col and not "NP1PTOT" in col) or col in ["PATNO", "INFODT", ]])).copy()
	p1pfeatures=[col for col in updrs1p.columns if "NP1" in col and not "NP1PTOT" in col]


	#updrs2=P2[P2["PAG_NAME"]=="NUPDRS1P2P"].copy()
	updrs2=P2.copy()
	updrs2=updrs2.sort_values(by=['INFODT', 'EVENT_ID'])
	updrs2=updrs2.drop(columns=updrs2.columns.difference([col for col in updrs2.columns if ("NP2" in col and not "NP2PTOT" in col) or col in ["PATNO", "INFODT",]])).copy()
	p2features=[col for col in updrs2.columns if "NP2" in col and not "NP2PTOT" in col]

	#updrs3=P3[P3["PAG_NAME"]=="NUPDRS3"].copy()
	updrs3=P3.copy()
	updrs3=updrs3[updrs3["PDSTATE"].isin(["ON", "OFF"])].copy()#evaluate only on medication
	updrs3["MED_STATE"] = updrs3["PDSTATE"]
	updrs3=updrs3[updrs3["MED_STATE"]==state].copy()
	updrs3=updrs3.sort_values(by=['INFODT', 'EVENT_ID'])
	target=updrs3.drop(columns=updrs3.columns.difference([col for col in updrs3.columns if col in ["PATNO", "INFODT", "EVENT_ID", "NHY"]])).copy()
	updrs3=updrs3.drop(columns=updrs3.columns.difference([col for col in updrs3.columns if ("NP3" in col and not "NP3TOT" in col) or col in ["PATNO", "INFODT",]])).copy()
	p3features=[col for col in updrs3.columns if ("NP3" in col and not "NP3TOT" in col)]


	#updrs4=P4[P4["PAG_NAME"]=="NUPDRS4"].copy()
	updrs4=P4.copy()
	updrs4=updrs4.sort_values(by=['INFODT', 'EVENT_ID'])
	updrs4=updrs4.drop(columns=updrs4.columns.difference([col for col in updrs4.columns if ("NP4" in col and not "NP4TOT" in col and not col in ["NP4DYSTNPCT", "NP4DYSTNNUM", "NP4DYSTNDEN", "NP4OFFPCT", "NP4OFFNUM", "NP4WDYSKDEN", "NP4WDYSKNUM", "NP4WDYSKPCT", "NP4OFFDEN"]) or col in ["PATNO", "INFODT"]])).copy()
	p4features=[col for col in updrs4.columns if "NP4" in col and not "NP4RTOT" in col]
	
	
	
	values=[101,0,1,2,3,4,np.nan]

	for col in p1features:
		assert updrs1[col].isin(values).all()
		updrs1[col]=updrs1[col].fillna(101)
	for col in p1pfeatures:
		assert updrs1p[col].isin(values).all()
		updrs1p[col]=updrs1p[col].fillna(101)
	for col in p2features:
		assert updrs2[col].isin(values).all()
		updrs2[col]=updrs2[col].fillna(101)
	for col in p3features:
		assert updrs3[col].isin(values).all()
		updrs3[col]=updrs3[col].fillna(101)
	for col in p4features:
		assert updrs4[col].isin(values).all()
		updrs4[col]=updrs4[col].fillna(101)
	
	#replace nans with 101
	dfs = [updrs1, updrs1p, updrs2, updrs3, updrs4]
	dfs_features=np.concatenate([p1features, p1pfeatures, p2features, p3features, p4features], axis=0)

	for i, d in enumerate(dfs):
		dfs[i] = d.copy()
		dfs[i]["INFODT"] = pd.to_datetime(dfs[i]["INFODT"], errors="coerce")
	keys = ["PATNO", "INFODT"]
	for i, d in enumerate(dfs):
		if d.duplicated(keys).any():
			dfs[i] = d.sort_values(keys).groupby(keys, as_index=False).first()
	all_clinical = reduce(lambda left, right: pd.merge(left, right, on=keys, how="outer"), dfs)
	feature_cols = all_clinical.columns.difference(keys)
	all_clinical[feature_cols] = all_clinical[feature_cols].fillna(101)
	
	pat_curves={}
	
	for patno in PD_PATNO:
		IND_DEMOG=DEMOG[DEMOG['PATNO']==patno]
		birth=pd.to_datetime(IND_DEMOG["BIRTHDT"], format='%m/%Y', errors='coerce')
		
		gender=IND_DEMOG["SEX"]

		# After you've created result (patient-specific already)
		keys = ["INFODT", "EVENT_ID"]

		
		all_clinical_patient = all_clinical[all_clinical["PATNO"] == patno].copy()
		datesall_clinical_patient=pd.to_datetime(all_clinical_patient["INFODT"], format='%m/%Y', errors='coerce').to_numpy()

		hy_patient=target[target["PATNO"]==patno]["NHY"].to_numpy()
		dateshy_patient=pd.to_datetime(target[target["PATNO"]==patno]["INFODT"], format='%m/%Y', errors='coerce').to_numpy()

		patno=str(patno)

		hy_patient_nearest = nearest_interp_dates(datesall_clinical_patient,dateshy_patient,hy_patient,fill_value=101)

	
		curve=Curve(patno, all_clinical_patient["INFODT"], birth, gender, dfs_features, np.concatenate([np.expand_dims(all_clinical_patient[col].to_numpy(), axis=-1) for col in dfs_features], axis=-1), hy_patient_nearest, ttol=1)
		if(curve is not None): pat_curves[patno]=curve
		print("I:",patno, "of", PD_PATNO.iloc[-1], end="\r")
	
	print("I: Successfully read progression curves.", end="\n")

	return pat_curves

class Transition(np.ndarray):

	n_features=65
	n_values=8
	end_seq=7*np.ones((1,65)).astype(np.int32)
	max_len=50
	#[0,1,2,3,4,5,nan,endseq]
	#[0,1,2,3,4,5,6,7,]
	
	def __new__(cls, patno : np.int32, duration : np.int32, start_age_weeks : np.int32, gender : np.int32, orig : State, dest : State, time : np.int32, measures : np.int32):

		obj=np.asarray([duration, start_age_weeks, gender, orig.x, dest.x, time]).view(cls).astype(np.int32)

		if(measures.shape[1]==1 and np.all(measures==101)): measures=np.repeat(measures,Transition.n_features,axis=1)

		obj.patno=patno
		obj.duration=duration
		obj.start_age_weeks=start_age_weeks
		obj.gender=gender
		obj.orig=orig.x
		obj.dest=dest.x
		obj.t=time
		obj.measures=measures
		obj.event=int(orig.x!=dest.x)
		obj.seq_len=obj.measures.shape[0]
		obj.measures=np.concatenate((obj.measures, Transition.end_seq), axis=0)#adjust seq len
		obj.measures[np.where(obj.measures==101)]=obj.n_values-2
		for s in range(len(State.states)):
			obj.measures[np.where(obj.measures==State.values[s])]=State.states[s]
		obj.measures=np.eye(obj.n_values)[obj.measures]
		
		obj.measures=np.concatenate((obj.measures,np.zeros((Transition.max_len,obj.n_features,obj.n_values,))[:Transition.max_len-obj.measures.shape[0]]), axis=0) #pad sequence with zeros
		
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.patno=getattr(obj, "patno", None)
		self.duration=getattr(obj, "duration", None)
		self.start_age_weeks=getattr(obj, "start_age_weeks", None)
		self.gender=getattr(obj, "gender", None)
		self.orig=getattr(obj, "orig", None)
		self.dest=getattr(obj, "dest", None)
		self.t=getattr(obj, "t", None)
		self.measures=getattr(obj, "measures", None)
		self.seq_len=getattr(obj, "seq_len", None)
		self.event=getattr(obj, "event", None)

ppmi_transitions=_GetTransitions(Transition, State, _get_curves)

class PPMI(_Dataset):

	get_dataset=get_ppmi
	curve=Curve
	state=State
	transition=Transition
	get_transitions=ppmi_transitions