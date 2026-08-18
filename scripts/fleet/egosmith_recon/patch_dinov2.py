"""Make uniception's DINOv2 fallback truly offline: torch.hub.load without a ref probes
github.com even on a full cache; on flaky pod egress that kills every slam worker."""
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
OLD = '''        except:  # Load from cache
            self.model = torch.hub.load("facebookresearch/dinov2", DINO_MODELS[self.with_registers][self.version])'''
NEW = '''        except:  # Load from the local hub cache (no network probe)
            import os as _os
            _cached = _os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
            self.model = torch.hub.load(_cached, DINO_MODELS[self.with_registers][self.version], source="local")'''
if "source=\"local\"" in src:
    print("already patched")
elif OLD in src:
    p.write_text(src.replace(OLD, NEW)); print("patched", p)
else:
    print("PATTERN NOT FOUND — manual review needed"); sys.exit(1)
