// Scene and Snapshot ingress, one rule per test, plus create/destroy loops for the leak sanitizer.
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"
#include "sscbirrt/mujoco/scene.hpp"
#include "sscbirrt/mujoco/snapshot.hpp"

using namespace sscbirrt;
using sscbirrt::mujoco::Attachment;
using sscbirrt::mujoco::Scene;
using sscbirrt::mujoco::Snapshot;

namespace {

const char* kXml = R"(
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="robot/base" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1" limited="true" range="-2 2"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/>
      <body name="robot/link1" pos="0.3 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/>
        <body name="robot/gripper/base" pos="0.3 0 0">
          <geom type="box" size="0.02 0.02 0.02"/>
          <body name="robot/gripper/finger" pos="0.03 0 0"><geom type="box" size="0.01 0.01 0.02"/></body>
        </body>
      </body>
    </body>
    <body name="can" pos="0.8 0 0.2"><freejoint/><geom type="cylinder" size="0.03 0.06"/></body>
    <body name="ball" pos="0 0.5 0.5"><joint type="ball"/><geom type="sphere" size="0.05"/></body>
    <body name="target" mocap="true" pos="1 1 1"><geom type="sphere" size="0.02" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>)";

std::vector<std::byte> mjb_of(const char* xml) {
  const auto path = std::filesystem::temp_directory_path() / "sscbirrt_test_scene.xml";
  {
    std::ofstream f(path);
    f << xml;
  }
  char err[1024] = {0};
  mjModel* m = mj_loadXML(path.string().c_str(), nullptr, err, sizeof err);
  if (!m) throw std::runtime_error(std::string("mj_loadXML: ") + err);
  std::vector<std::byte> out(static_cast<std::size_t>(mj_sizeModel(m)));
  mj_saveModel(m, nullptr, out.data(), static_cast<int>(out.size()));
  mj_deleteModel(m);
  return out;
}

const std::vector<std::byte>& mjb() {
  static const std::vector<std::byte> bytes = mjb_of(kXml);
  return bytes;
}

Snapshot rest(const Scene& s) {
  Snapshot snap;
  snap.qpos.assign(static_cast<std::size_t>(s.nq()), 0.0);
  // the can's free joint: position then unit quaternion
  const int can = s.body_id("can");
  (void)can;
  // qpos layout: j0, j1, can (7), ball (4). Set unit quaternions.
  snap.qpos[2 + 3] = 1.0;  // can quat w
  snap.qpos[2 + 7] = 1.0;  // ball quat w
  snap.mocap_pos = {1.0, 1.0, 1.0};
  snap.mocap_quat = {1.0, 0.0, 0.0, 0.0};
  return snap;
}

}  // namespace

TEST(versions_agree_and_provenance_is_recorded) {
  CHECK(mujoco::compiled_mujoco_version() == mujoco::loaded_mujoco_version());
  Scene s(mjb(), {"j0", "j1"}, {}, 42);
  CHECK(s.provenance().mujoco_version == mj_versionString());
  CHECK(s.provenance().model_signature == 42);  // as passed; MJB does not carry the signature
  CHECK(s.provenance().mjb_sha256.size() == 64);
  Scene t(mjb(), {"j0", "j1"}, {}, 42);
  CHECK(t.provenance().mjb_sha256 == s.provenance().mjb_sha256 && t.provenance().model_signature == s.provenance().model_signature);
}

TEST(scene_resolves_joints_limits_and_arm_bodies) {
  Scene s(mjb(), {"j0", "j1"});
  CHECK(s.dof() == 2 && s.qpos_addresses() == (std::vector<int>{0, 1}));
  CHECK(s.lower()[0] == -2.0 && s.upper()[0] == 2.0);
  CHECK(std::isinf(s.lower()[1]) && std::isinf(s.upper()[1]));  // unlimited hinge reports ±infinity (#107)
  for (const char* name : {"robot/base", "robot/link1", "robot/gripper/base", "robot/gripper/finger"}) CHECK(s.is_arm_body(s.body_id(name)));
  CHECK(!s.is_arm_body(s.body_id("can")) && !s.is_arm_body(s.body_id("floor") ));  // floor is a world geom: body 0
  CHECK(s.body_has_free_joint(s.body_id("can")) && !s.body_has_free_joint(s.body_id("ball")) && !s.body_has_free_joint(s.body_id("robot/base")));
  CHECK(s.subtree(s.body_id("robot/gripper/base")).size() == 2);
  CHECK(s.nmocap() == 1);
  Scene with_extra(mjb(), {"j0"}, {"can"});
  CHECK(with_extra.is_arm_body(with_extra.body_id("can")));
}

TEST(scene_rejects_malformed_input_one_rule_at_a_time) {
  CHECK_THROWS(Scene(std::vector<std::byte>{}, {"j0"}), std::invalid_argument);
  std::vector<std::byte> junk(100, std::byte{0x2a});
  CHECK_THROWS(Scene(junk, {"j0"}), std::invalid_argument);
  std::vector<std::byte> truncated(mjb().begin(), mjb().begin() + static_cast<std::ptrdiff_t>(mjb().size() / 2));
  CHECK_THROWS(Scene(truncated, {"j0"}), std::invalid_argument);
  CHECK_THROWS(Scene(mjb(), {}), std::invalid_argument);
  CHECK_THROWS(Scene(mjb(), {"j0", "j0"}), std::invalid_argument);
  CHECK_THROWS(Scene(mjb(), {"nope"}), std::invalid_argument);
  CHECK_THROWS(Scene(mjb(), {"j0"}, {"nope"}), std::invalid_argument);
  // The ball body's joint is unnamed; name-less joints cannot be controlled, and a free joint is refused by type.
  Scene probe(mjb(), {"j0"});
  const int can_joint = probe.model()->body_jntadr[probe.body_id("can")];
  (void)can_joint;
}

TEST(snapshot_validates_and_hashes) {
  Scene s(mjb(), {"j0", "j1"});
  Snapshot a = rest(s);
  a.validate(s);
  CHECK(a.sha256.size() == 64);
  Snapshot b = rest(s);
  b.validate(s);
  CHECK(a.sha256 == b.sha256);
  b.qpos[0] = 0.5;
  b.validate(s);
  CHECK(a.sha256 != b.sha256);

  Snapshot bad = rest(s);
  bad.qpos.pop_back();
  CHECK_THROWS(bad.validate(s), std::invalid_argument);
  bad = rest(s);
  bad.qpos[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(bad.validate(s), std::invalid_argument);
  bad = rest(s);
  bad.mocap_quat = {2.0, 0.0, 0.0, 0.0};
  CHECK_THROWS(bad.validate(s), std::invalid_argument);
  bad = rest(s);
  bad.mocap_pos.pop_back();
  CHECK_THROWS(bad.validate(s), std::invalid_argument);
}

TEST(attachments_validate_and_order_does_not_matter) {
  Scene s(mjb(), {"j0", "j1"});
  const int can = s.body_id("can"), grip = s.body_id("robot/gripper/finger"), base = s.body_id("robot/gripper/base");
  Attachment ok;
  ok.object_body = can;
  ok.gripper_body = grip;
  ok.allowed_bodies = {grip, base, grip};  // duplicates are removed
  Snapshot a = rest(s);
  a.attachments = {ok};
  a.validate(s);
  CHECK(a.attachments[0].allowed_bodies == (std::vector<int>{base, grip}));

  Attachment bad = ok;
  bad.object_body = s.body_id("ball");  // ball joint, not free
  Snapshot t = rest(s);
  t.attachments = {bad};
  CHECK_THROWS(t.validate(s), std::invalid_argument);
  bad = ok;
  bad.gripper_body = s.body_id("can");  // not an arm body
  t = rest(s);
  t.attachments = {bad};
  CHECK_THROWS(t.validate(s), std::invalid_argument);
  bad = ok;
  bad.T_gripper_object.at(0, 0) = 2.0;  // not a rotation
  t = rest(s);
  t.attachments = {bad};
  CHECK_THROWS(t.validate(s), std::invalid_argument);
  bad = ok;
  bad.allowed_bodies = {999};
  t = rest(s);
  t.attachments = {bad};
  CHECK_THROWS(t.validate(s), std::invalid_argument);
  t = rest(s);
  t.attachments = {ok, ok};  // attached twice
  CHECK_THROWS(t.validate(s), std::invalid_argument);
}

TEST(create_and_destroy_in_a_loop_leaks_nothing) {
  for (int i = 0; i < 20; ++i) {
    Scene s(mjb(), {"j0", "j1"});
    Snapshot snap = rest(s);
    snap.validate(s);
    CHECK(s.dof() == 2);
  }
  for (int i = 0; i < 20; ++i) {
    try {
      Scene s(mjb(), {"j0", "nope"});
    } catch (const std::invalid_argument&) {
    }
  }
}

HARNESS_MAIN()
