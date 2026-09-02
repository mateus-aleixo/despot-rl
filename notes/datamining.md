# Datamining Despot's Game

The shipped balance tables are AES-... no: **Rijndael-256** encrypted, gzipped, and
concatenated. This note records how they were opened, so the extraction can be
re-run after a game patch.

## Build

- Unity 2020.3.49f1, IL2CPP, x64.
- Install: `C:\Program Files (x86)\Steam\steamapps\common\Despot's Game`
- Symbol names are partially obfuscated (Beebyte-style 27-28 char identifiers),
  but the `DataLoaders` namespace, the `M_*` model classes and `StringCipher`
  all survive in the clear.

## Where the data lives

Five `TextAsset`s inside the Addressables bundles under
`Despot's Game_Data/StreamingAssets/aa/StandaloneWindows64/`:

| TextAsset | Encrypted size | Decoded |
|---|---|---|
| `EncryptedMainGroup` | 121 376 B | 1 170 670 B, 95 json files |
| `EncryptedMainTasksGroup` | 53 472 B | 521 379 B, 50 json files |
| `EncryptedDLCGroup` | 140 864 B | 1 359 542 B, 90 json files |
| `EncryptedLocalizationsGroup` | 1 117 920 B | 4 157 929 B, 109 json files |
| `metadata` | 3 232 B | 26 892 B |

## The scheme

`EncryptedAssetProvider.InternalOp.ActionComplete` does:

```
StringCipher.Decrypt(bytes, "<passphrase>")  ->  FS.Unzip  ->  new TextAsset(...)
```

The passphrase is a plain 32-character IL2CPP string literal, held out of this
repository: the tools read it from `DESPOT_PASSPHRASE`, and it is recovered from
your own installation by disassembling `EncryptedAssetProvider` (see "Dead ends"
below for the sweep that does not find it, and the one that does).

`StringCipher` is the widely copy-pasted StackOverflow class:

- `Keysize = 256`, `DerivationIterations = 1000`
- payload layout `salt[32] || iv[32] || ciphertext`
- key = `PBKDF2-HMAC-SHA1(passphrase, salt, 1000, 32)`
- **RijndaelManaged with BlockSize = 256**, CBC, PKCS7

The 256-bit block is the trap: this is Rijndael-256, not AES, so no mainstream
crypto library will decrypt it. `tools/rijndael256.py` implements it; its core is
verified against pycryptodome by running it at Nb=4, where Rijndael is AES.

`FS.Unzip` is gzip, not zip. The decompressed payload is a flat stream of
`path\n<single-line json>\n` records.

## Dead ends, recorded so they are not repeated

- The passphrase *is* a plain IL2CPP string literal, but a literal sweep filtered
  on "decrypted head looks like text" misses it: the plaintext is gzip, so the
  first bytes are `1f 8b 08`. Filter on entropy or on known magic bytes instead.
- The key is not in `fieldAndParameterDefaultValueData`; sliding a 16/24/32-byte
  window over that blob finds nothing, because there is no `byte[]` key at all.
- `Il2CppDumper` finds no direct `call` site for `StringCipher.Decrypt`; IL2CPP
  dispatches it indirectly. Disassembling the *caller* found by name
  (`EncryptedAssetProvider`) is what works.

**`tools/xrefs.py` used to be wrong, and its "0 call sites" was believable.** It
scanned only the `.text` section, which is 3 MB of runtime glue; every generated
game method lives in the 32 MB `il2cpp` section, so it reported no callers for
methods that plainly have them. It now scans both, and direct calls are the
common case: `C_Team.GetExperience` has five callers and one of them,
`C_Unit.Die`, is the whole answer to "where does experience come from". Treat an
empty result from a tool as a result about the tool until it has been shown a
case it should find.

## Pipeline

```
tools/dump_textassets.py    # UnityPy: pull the 5 encrypted TextAssets out of the bundles
tools/decrypt_blobs.py      # Rijndael-256 + PBKDF2 + gunzip
tools/split_gamedata.py     # split path\njson records into data/extracted/json/
```

Supporting tools: `probe_assets.py`, `unit_roster.py` (prefab names),
`adisasm.py` (annotated disassembler using Il2CppDumper's `script.json`),
`xrefs.py`, `symmap.py`.

## What came out

`data/extracted/json/EncryptedMainGroup/DB/`:

- `Units.json` - 168 rows, 111 classes x levels 1-5. Fields: `Health`, `Damage`,
  `AttackSpeed`, `Range`, `Armor`, `Resistance`, `Speed`, `Mana`, `Power`,
  `ExpReward`, `GoldReward`, `Skill1`..`Skill8`.
- `Items.json` - 56 weapons over 11 player classes (Warrior, Medic, Shooter,
  Tank, Thrower, Monk, Scientist, Dodger, Cultist, Mage, Plant). Base stats plus
  `*PerLevel` scaling, `Cost`, `Quality`, `Power`.
- `Skills.json` - 142 skills, each up to 10 named parameters.
- `EnemyPacks.json` - 212 packs: class, count range, room type, zone.
- `Levels.json` - 12 levels: biom, room counts, `PowerPerRoom`, food and item
  multipliers, shop counts.
- `Game.json` - starting gold and food, `ExperienceForLevels`, team setup.

Chip variants (Easy/Hard/Crazy/Hunger/WithoutFood), Arcade, KingOfTheHill and
per-task overrides all ship as separate override files under the same tree.

## Prefabs, behaviour trees, pathfinding graphs

The balance JSON is only half the picture. Full-fidelity combat also needs the
per-unit steering parameters, the AI graphs and the room navmeshes.

**A\* graphs.** The `graph*` TextAssets in `sharedassets3.assets` are A\*
Pathfinding Project 4.2.14 cache files, and they are plain zips: open with
`zipfile`, no tooling needed. Each holds three `Pathfinding.GridGraph`s that
differ only in collision diameter (1.99 / 2.99 / 3.99), i.e. one walkability grid
per unit size class. 2D mode, `nodeSize: 6`. The file name is the room shape:

| Asset | unclampedSize | grid |
|---|---|---|
| `graph` | 492 x 222 | 82 x 37 |
| `graph-horizontal` | 1068 x 222 | 178 x 37 |
| `graph-vertical-3` | 492 x 942 | 82 x 157 |
| `graph-angle-top-left` | 1068 x 582 | 178 x 97 |

**MonoBehaviour fields.** IL2CPP builds ship no type trees, so UnityPy returns
raw bytes for prefab components. AssetRipper 2.0.0 resolves them. It is a local
web app, but it is scriptable: run it `--headless --port <n>`, then
`POST /LoadFile` (form field `Path`) and `POST /Export/UnityProject`. The API is
described at `/openapi.json`; `/Assets/Json?Path=...` reads a single asset if a
full export is overkill.

Exporting the unit bundle `7443a426cc273843ed5743fb31c72054.bundle` gives 269
prefabs plus 54 `NC-*.asset` behaviour trees. Sample values off `Swordsman`:

- `UnitMovement`: `radius: 8`, `speed: 100`, `pickNextWaypointDist: 12`
- `RVOController`: `radius: 6`, `agentTimeHorizon: 0.5`,
  `obstacleTimeHorizon: 0.5`, `maxNeighbours: 10`, `wallAvoidForce: 1`,
  `wallAvoidFalloff: 1`, `priority: 0.5`, `lockWhenNotMoving: 1`
- `Seeker`: `graphMask: -7`
- `V_Unit`: `ranged`, projectile reference, sound ids

Note `UnitMovement.radius` (8) and `RVOController.radius` (6) differ; they feed
different systems (pathfinding clearance vs local avoidance).

**Behaviour trees** deserialize straight out of `_serializedGraph` as NodeCanvas
JSON: typed nodes (`Selector`, `Sequencer`, `ConditionNode`, `ActionNode`) with
`$type` naming the concrete condition or action (`ShouldDie`, `Die`,
`WaitForAnimation`, `BecomeCorpse`, `Spawn`, ...) and `$id` cross-references. The
semantics of each action live in `dump.cs` / `DummyDll`.

`tools/read_prefab.py` prints a ripped prefab's components with script names
resolved from the `.cs.meta` GUIDs.

## Ruleset layering

`metadata.json` (the decoded `metadata` TextAsset) carries a `Modes` map that is
the authoritative load order. Layers apply in sequence:

    Common  ->  <Mode>  ->  <Mode>.Chips[<chip>]  ->  ...WithoutFood

Modes are `Common`, `Default`, `KingOfTheHill`, `Arcade`, `Tasks`; chips under
`Default` are `default`, `easy`, `hard`, and others. Each entry is
`LogicalName: "path/to.json?strategy"`, where a missing strategy means
Newtonsoft's `JObject.Merge` (recurse into objects, replace everything else).
Observed strategies: `replace`, `mergeGrid`, `mergeByID`,
`mergeByMutationAndLevel`; `Loader` also holds `remover` and `cloner`.

The target ruleset (Default mode, `default` chip, with food) therefore replaces
`EnemyPacks` with `DB/Chips/Default/EnemyPacksShortened.json` and
`SimpleMutations` with `MutationsDefaultSpecial.json`, and merges the mutation
grid.

`sim/data.py` reproduces this. `mergeGrid` and `mergeByMutationAndLevel` are not
implemented; the loader **raises** on them rather than merging wrongly, so
mutations must wait for those two strategies. Everything the battle layer needs
(Units, Items, Skills, EnemyPacks, Levels, Rooms, RoomLayouts, Game) resolves
cleanly.

## Mechanics confirmed from the data

- Player humans are the `Novice` class; the equipped item supplies the real class
  (Warrior, Medic, Shooter, Tank, Thrower, Monk, Scientist, Dodger, Cultist,
  Mage, Plant) and adds its stats on top. `Swordsman` and friends are prefab
  names, not `Units.json` rows.
- `Units.json` `Level` 1-5 is the unit upgrade level; items scale separately via
  their `*PerLevel` fields.
- Room layouts: 77 blocks, almost all a 7 x 20 grid. The player zone is a 7 x 7
  block of `p` cells with corner markers `p:1`..`p:4`; enemies occupy `e1`/`e2`
  zones; `s` marks the shop/door cell. This grid is the pre-fight placement
  surface.

## The dialog table is not under `DB/`

`EncryptedMainGroup` holds five files outside its `DB/` subtree:
`achievements.json`, `metadata.json`, `empty-array.json`, `empty-layouts.csv`
and **`dialogs.json`**, the 115 KB level-entry dialog table. Every earlier
search treated `EncryptedMainGroup/DB` as the whole group, which is why the
table read as missing for as long as it did. When something is described as a
table and is not under `DB/`, list the group root before concluding it ships
somewhere else.

Its strings are all localization keys, so it is only readable joined against
`EncryptedLocalizationsGroup/Languages/en.json`. `tools/show_dialog.py` does
that join. The decoded schema is in `notes/reference-sim.md`.
