import uproot   
import awkward as ak
import numpy as np
import pandas as pd
from utils.vectors import vec4

def add_multiplicty(events):
    events = ak.with_field(events, ak.num(events.Muon), "nMuon")
    events = ak.with_field(events, ak.num(events.SV), "nSV")
    events = ak.with_field(events, ak.num(events.SVOverlap), "nSVOverlap")

    return events

def trigger_filter(events):
    dimuon_trigger = (events["L1_DoubleMu_15_7"] | events["L1_DoubleMu4p5er2p0_SQ_OS_Mass_Min7"] | events["L1_DoubleMu4p5_SQ_OS_dR_Max1p2"])
    events = events[dimuon_trigger]

    return events

# This loop takes a while, probably would want to optimize it
def get_sv_mass(events):
    sv_mass = ak.ones_like(events.SV.lxy) * -1
    muons = events.Muon
    muon_pairs = ak.combinations(muons, 2, fields=["muon_i", "muon_j"])

    os_mask = muon_pairs.muon_i.ch != muon_pairs.muon_j.ch 
    same_vtx_mask = (muon_pairs.muon_i.bestAssocSVIdx == muon_pairs.muon_j.bestAssocSVIdx)
    valid_vtx_mask = muon_pairs.muon_i.bestAssocSVIdx > -1
    mask = np.logical_and(os_mask, same_vtx_mask)
    mask = np.logical_and(mask, valid_vtx_mask)

    muon_pairs = muon_pairs[mask]

    sv_mass = ak.ArrayBuilder()

    for i_event in range(len(events)):
        events_i = events[i_event]
        muon_pairs_i = muon_pairs[i_event]
        svs_i = events_i.SV
        svs_mass_i = np.full(len(svs_i), -1.0)

        if len(muon_pairs_i) > 0:
            for muon_pair in muon_pairs_i:
                muon_i = muon_pair.muon_i
                muon_j = muon_pair.muon_j

                muon_i_vec = vec4()
                muon_i_vec.setPtEtaPhiM(muon_i.pt, muon_i.eta, muon_i.phiCorr, 0.10566)
                muon_j_vec = vec4()
                muon_j_vec.setPtEtaPhiM(muon_j.pt, muon_j.eta, muon_j.phiCorr, 0.10566)
                dimuon = muon_i_vec + muon_j_vec
                
                sv_idx = muon_i.bestAssocSVIdx

                for i_sv in range(len(svs_i)):
                    if (svs_i.index[i_sv] == sv_idx):
                        svs_mass_i[i_sv] = dimuon.m
                        break

        sv_mass.append(svs_mass_i)

    sv = events.SV
    sv = ak.with_field(sv, sv_mass, "mass")
    events = ak.with_field(events, sv, "SV")

    return events
    