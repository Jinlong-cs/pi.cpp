# Revision 7 Legacy Fixture Gap

Read-only audit of the prior V18 fixture on `agx-test`:

- Source: `/root/wujinlong/nvme/picpp_walloss_20260818/revision3_export/pytorch/stage_tensors_explicitcache.pt`
- It contains the required public stage dictionaries, 73 prefill outputs and
  72 KV tensors.
- It has no `trajectory` dictionary, so `x1..x10` are not available as saved
  PyTorch references.
- The 5090 Rev5 `fixtures/` directory is empty; its copied inbound V18 fixture
  is the same single-fixture artifact and is not a held-out split.
- No `/home/zem/wujinlong/picpp_walloss_20260818/revision7_export/` directory
  existed during the 2026-08-21 audit.

This is a coverage gap, not a new model result. The fixture cannot satisfy the
Revision 7 reference-completeness gate or authorize engine construction.
