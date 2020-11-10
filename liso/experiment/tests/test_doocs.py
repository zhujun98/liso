import unittest

import numpy as np

from pydantic import ValidationError

from liso.experiment.doocs import (
    doocs_channels, DoocsChannel,
    BoolDoocsChannel,
    Int64DoocsChannel, UInt64DoocsChannel,
    Int32DoocsChannel, UInt32DoocsChannel,
    Int16DoocsChannel, UInt16DoocsChannel,
    Float32DoocsChannel,
    Float64DoocsChannel,
    ImageDoocsChannel,
)


class TestDoocs(unittest.TestCase):
    def testDoocsChannel(self):
        ch = DoocsChannel(address="A/B/C/D")

    def testInt64DoocsChannel(self):
        self.assertEqual(doocs_channels.LONG, Int64DoocsChannel)
        self.assertEqual(doocs_channels.INT64, Int64DoocsChannel)

        with self.assertRaises(ValidationError):
            Int64DoocsChannel(address="XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1/")
        with self.assertRaises(ValidationError):
            Int64DoocsChannel(address="XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/")
        with self.assertRaises(ValidationError):
            Int64DoocsChannel(address="XFEL.RF/LLRF.SUMVOLTAGE_CTRL//SUMVOLTAGE.CHIRP.SP.1")

        ch = Int64DoocsChannel(address="A/B/C/D", value=1)
        self.assertEqual("A/B/C/D", ch.address)
        self.assertEqual(1, ch.value)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = 1.1
            with self.assertRaises(ValidationError):
                ch.value = True

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Address', 'type': 'string'}, schema['address'])
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '<i8'}, schema['value'])

        with self.subTest("Test value schema"):
            self.assertDictEqual({'default': 0, 'type': '<i8'}, ch.value_schema())

    def testUInt64DoocsChannel(self):
        self.assertEqual(doocs_channels.ULONG, UInt64DoocsChannel)
        self.assertEqual(doocs_channels.UINT64, UInt64DoocsChannel)

        ch = UInt64DoocsChannel(address="A/B/C/D", value=0)
        self.assertEqual(False, ch.value)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = -1

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'minimum': 0, 'type': '<u8'}, schema['value'])

    def testInt32DoocsChannel(self):
        self.assertEqual(doocs_channels.INT, Int32DoocsChannel)
        self.assertEqual(doocs_channels.INT32, Int32DoocsChannel)

        ch = Int32DoocsChannel(address="A/B/C/D", value=0)
        self.assertEqual(False, ch.value)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = 1.1

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '<i4',
                 'minimum': -2 ** 31, 'maximum': 2 ** 31 - 1},
                schema['value'])

    def testUInt32DoocsChannel(self):
        self.assertEqual(doocs_channels.UINT, UInt32DoocsChannel)
        self.assertEqual(doocs_channels.UINT32, UInt32DoocsChannel)

        ch = UInt32DoocsChannel(address="A/B/C/D", value=0)
        self.assertEqual(False, ch.value)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = -1

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '<u4',
                 'minimum': 0, 'maximum': 2 ** 32 - 1},
                schema['value'])

    def testInt16DoocsChannel(self):
        self.assertEqual(doocs_channels.INT16, Int16DoocsChannel)

        ch = Int16DoocsChannel(address="A/B/C/D", value=0)
        self.assertEqual(False, ch.value)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = 1.1

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '<i2',
                 'minimum': -32768, 'maximum': 32767},
                schema['value'])

    def testUInt16DoocsChannel(self):
        self.assertEqual(doocs_channels.UINT16, UInt16DoocsChannel)

        ch = UInt16DoocsChannel(address="A/B/C/D", value=0)
        self.assertEqual(False, ch.value)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = -1

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '<u2',
                 'minimum': 0, 'maximum': 65535},
                schema['value'])

    def testBoolDoocsChannel(self):
        self.assertEqual(doocs_channels.BOOL, BoolDoocsChannel)

        ch = BoolDoocsChannel(address="A/B/C/D", value=True)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = 0

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '|b1'}, schema['value'])

    def testFloat64DoocsChannel(self):
        self.assertEqual(doocs_channels.DOUBLE, Float64DoocsChannel)
        self.assertEqual(doocs_channels.FLOAT64, Float64DoocsChannel)

        ch = Float64DoocsChannel(address="A/B/C/D", value=1.)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = 1

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '<f8'}, schema['value'])

    def testFloat32DoocsChannel(self):
        self.assertEqual(doocs_channels.FLOAT, Float32DoocsChannel)
        self.assertEqual(doocs_channels.FLOAT32, Float32DoocsChannel)

        ch = Float32DoocsChannel(address="A/B/C/D", value=1.)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaises(ValidationError):
                ch.value = 1

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Value', 'default': 0, 'type': '<f4',
                 'maximum': np.finfo(np.float32).max, 'minimum': np.finfo(np.float32).min},
                schema['value'])

    def testImageChannel(self):
        self.assertEqual(doocs_channels.IMAGE, ImageDoocsChannel)

        for k, v in {"int": "<i8",
                     "uint16": "<u2",
                     "float": "<f8",
                     "float32": "<f4",
                     "f8": "<f8"}.items():
            ch = ImageDoocsChannel(address="A/B/C/D", shape=(2, 2), dtype=k)
            self.assertEqual(v, ch.dtype)

        with self.subTest("Test shape and dtype are mandate"):
            with self.assertRaisesRegex(ValidationError, 'dtype'):
                ImageDoocsChannel(address="A/B/C/D", shape=(2, 2))
            with self.assertRaisesRegex(ValidationError, 'shape'):
                ImageDoocsChannel(address="A/B/C/D", dtype="<i8")

        ch = ImageDoocsChannel(address="A/B/C/D", shape=(2, 2), dtype="<i8")
        self.assertTupleEqual((2, 2), ch.shape)
        np.testing.assert_array_equal(np.zeros((2, 2), dtype=np.int64), ch.value)
        with self.subTest("Test validation is on for assignment"):
            with self.assertRaisesRegex(ValidationError, "numpy.ndarray"):
                ch.value = 1
            with self.assertRaisesRegex(ValidationError, "dtype"):
                ch.value = np.ones((2, 2), dtype=np.uint64)
            with self.assertRaisesRegex(ValidationError, "shape"):
                ch.value = np.ones((3, 2), dtype=np.int64)
        ch.value = 10 * np.ones((2, 2), dtype=np.int64)
        np.testing.assert_array_equal(np.array([[10, 10], [10, 10]], dtype=np.int64), ch.value)

        with self.subTest("Test schema"):
            schema = ch.schema()['properties']
            self.assertDictEqual(
                {'title': 'Address', 'type': 'string'}, schema['address'])
            self.assertDictEqual(
                {'title': 'Shape', 'type': 'array',
                 'items': [{'type': 'integer'}, {'type': 'integer'}]}, schema['shape'])
            self.assertDictEqual(
                {'title': 'Dtype', 'type': 'string'}, schema['dtype'])
            self.assertDictEqual(
                {'title': 'Value', 'type': 'NDArray'}, schema['value'])

        with self.subTest("Test value schema"):
            self.assertDictEqual({'type': 'NDArray', 'shape': (2, 2), 'dtype': '<i8'},
                                 ch.value_schema())
